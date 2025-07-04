# CodeConnect is a social networking and goal tracking application using Flask
# This application allows users to manage profiles, set goals, connect with friends,
# and communicate via messages and group chats in real time, with an admin panel for user management 
# as well as accountability partner functionalities.

#library imports and modules
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import json
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import datetime, timezone
import traceback
import uuid
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "supersecretkey" 

# SQLite database configuration to store websites data
basedir = os.path.abspath(os.path.dirname(__file__)) 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'flask_app.db')  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
UPLOAD_FOLDER = os.path.join(basedir, 'static', 'Uploads')  # Directory for message file uploads
PROFILE_PHOTO_FOLDER = os.path.join(basedir, 'static', 'profile_photos')  # Directory for profile photos
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROFILE_PHOTO_FOLDER'] = PROFILE_PHOTO_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # create if folder doesn't exist to store files
os.makedirs(app.config['PROFILE_PHOTO_FOLDER'], exist_ok=True)  # Create folder if doesn't exist to store profile photos


db = SQLAlchemy(app)


# Hardcoded admin credentials for simplicity (in production use only), list for grouping all associated details to admin
# constant as does not change
ADMIN_CREDENTIALS = {
    'email': 'admin@example.com',
    'username': 'admin',
    'password_hash': generate_password_hash('adminpassword')  
}

# File upload constraints for profile photos
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # list for allowed file extensions, allows iteration for checking 
MAX_PROFILE_PHOTO_SIZE = 2 * 1024 * 1024  

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Association tables for many-to-many relationships
dismissed_suggestions = db.Table('dismissed_suggestions',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('suggested_user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True)
)

friends = db.Table('friends',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('friend_id', db.Integer, db.ForeignKey('user.id'), primary_key=True)
)

accountability_partners = db.Table('accountability_partners',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('partner_id', db.Integer, db.ForeignKey('user.id'), primary_key=True)
)

group_members = db.Table('group_members',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('group_id', db.Integer, db.ForeignKey('group.id'), primary_key=True)
)

# FriendRequest model for managing friend requests
class FriendRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # ID of the request to send from
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # ID of the request to send to
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))  
    status = db.Column(db.String(20), default='pending')  # status to track if pending, accepted, rejected

    sender = db.relationship('User', foreign_keys=[sender_id], backref=db.backref('sent_requests', cascade='all, delete-orphan'))
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref=db.backref('received_requests', cascade='all, delete-orphan'))

# model for users goals
class Goal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  
    title = db.Column(db.String(100), nullable=False)  
    description = db.Column(db.Text, nullable=True) 
    difficulty = db.Column(db.String(20), nullable=False)  # easy/medium/hard - changes amount of points
    points = db.Column(db.Integer, nullable=False) 
    status = db.Column(db.String(20), default='not_started')  # Status to track if not_started, in_progress, completed
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))  

    user = db.relationship('User', backref=db.backref('goals', cascade='all, delete-orphan'))

# Group model for group chats
class Group(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # Group name
    creator_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))  

    creator = db.relationship('User', backref=db.backref('created_groups', cascade='all, delete-orphan'))
    members = db.relationship('User', secondary=group_members, backref='groups')  # Many-to-many relationship with users

# GroupMessage model for messages in group chats
class GroupMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'), nullable=False)  
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  
    content = db.Column(db.Text, nullable=True)  # Message text content
    file_path = db.Column(db.String(300), nullable=True)  # Path to uploaded file
    file_size = db.Column(db.String(20), nullable=True)  
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc)) 
    status = db.Column(db.String(20), default='delivered')  # Status to track if delivered, seen

    sender = db.relationship('User', backref=db.backref('group_messages', cascade='all, delete-orphan'))
    group = db.relationship('Group', backref=db.backref('messages', cascade='all, delete-orphan'))

# User model for user data
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False) 
    email = db.Column(db.String(100), unique=True, nullable=False)  
    password = db.Column(db.String(200), nullable=False)  # Hashed password
    profile_photo = db.Column(db.String(300), nullable=True)  # Path to profile photo
    points = db.Column(db.Integer, default=0)  # User's points for completed goals
    tags = db.Column(db.Text, nullable=True, default="[]")  # JSON string of user tags for friend suggestions - can track lots of tags for each user

    # Many-to-many relationships
    friends = db.relationship(
        'User', 
        secondary=friends,
        primaryjoin=(id == friends.c.user_id),
        secondaryjoin=(id == friends.c.friend_id),
        backref=db.backref('friended_by', lazy='dynamic'),
        lazy='dynamic'
    )

    accountability_partners = db.relationship(
        'User',
        secondary=accountability_partners,
        primaryjoin=(id == accountability_partners.c.user_id),
        secondaryjoin=(id == accountability_partners.c.partner_id),
        backref=db.backref('accountability_partners_of', lazy='dynamic'),
        lazy='dynamic'
    )

    dismissed_suggestions = db.relationship(
        'User',
        secondary=dismissed_suggestions,
        primaryjoin=(id == dismissed_suggestions.c.user_id),
        secondaryjoin=(id == dismissed_suggestions.c.suggested_user_id),
        backref=db.backref('dismissed_by', lazy='dynamic'),
        lazy='dynamic'
    )

    # Add a friend to the user's friend list
    def add_friend(self, user):
        if not self.is_friend(user):
            self.friends.append(user)
            user.friends.append(self)
            return self
            
    # Remove a friend from the user's friend list
    def remove_friend(self, user):
        if self.is_friend(user):
            self.friends.remove(user)
            user.friends.remove(self)
            return self
            
    # Check if a user is a friend
    def is_friend(self, user):
        return self.friends.filter(friends.c.friend_id == user.id).count() > 0

    # Add an accountability partner
    def add_accountability_partner(self, user):
        if not self.is_accountability_partner(user):
            self.accountability_partners.append(user)
            user.accountability_partners.append(self)
            return self

    # Remove an accountability partner
    def remove_accountability_partner(self, user):
        if self.is_accountability_partner(user):
            self.accountability_partners.remove(user)
            user.accountability_partners.remove(self)
            return self

    # Check if a user is an accountability partner
    def is_accountability_partner(self, user):
        return self.accountability_partners.filter(accountability_partners.c.partner_id == user.id).count() > 0

    # Dismiss a friend suggestion
    def dismiss_suggestion(self, user):
        if not self.has_dismissed_suggestion(user):
            self.dismissed_suggestions.append(user)
            return self

    # Check if a suggestion has been dismissed
    def has_dismissed_suggestion(self, user):
        return self.dismissed_suggestions.filter(dismissed_suggestions.c.suggested_user_id == user.id).count() > 0

    # Check if user is a member of a group
    def is_group_member(self, group):
        return group in self.groups

# Message model for direct messages between users
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) 
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  
    content = db.Column(db.Text, nullable=True)  
    file_path = db.Column(db.String(300), nullable=True)  # Path to uploaded file
    file_size = db.Column(db.String(20), nullable=True) 
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))  
    status = db.Column(db.String(20), default='delivered')  

    sender = db.relationship('User', foreign_keys=[sender_id], backref=db.backref('sent_messages', cascade='all, delete-orphan'))
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref=db.backref('received_messages', cascade='all, delete-orphan'))

# Get the current logged-in user
def get_current_user():
    if 'user_email' in session:
        return User.query.filter_by(email=session['user_email']).first()
    return None

# Check if the current user is an admin
def is_admin_user():
    current_user = get_current_user()
    if not current_user:
        return False
    return (current_user.email == ADMIN_CREDENTIALS['email'] and
            current_user.name == ADMIN_CREDENTIALS['username'] and
            check_password_hash(current_user.password, 'adminpassword'))

# Generate TF-IDF vectors for user tags for friend suggestion similarity
def get_tag_vectors(users):
    tag_strings = [' '.join(json.loads(user.tags)) for user in users] #json to load tags
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tag_strings)
    return tfidf_matrix

# Home page route
@app.route('/')
def home():
    current_user = get_current_user()
    user_name = session.get('user_name', None)
    points = current_user.points if current_user else 0
    clear_form_state = session.pop('clear_form_state', False)
    return render_template('home.html', user_name=user_name, points=points, current_user=current_user, is_admin_user=is_admin_user, clear_form_state=clear_form_state)

# About page route
@app.route('/about')
def about():
    if 'user_email' not in session:
        flash("Please sign in to explore more about our community!", 'info')
        return redirect(url_for('home'))
    current_user = get_current_user()
    points = current_user.points if current_user else 0
    return render_template('about.html', points=points, is_admin_user=is_admin_user)

# Chat page route
@app.route('/chat')
def chat():
    if 'user_email' not in session:
        flash("Please sign in to connect with friends in the chat!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    points = current_user.points if current_user else 0
    return render_template('chat.html', current_user=current_user, active_friend=None, active_group=None, points=points, is_admin_user=is_admin_user)

# Goals page route
@app.route('/goals')
def goals():
    if 'user_email' not in session:
        flash("Please sign in to set and track your goals!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    points = current_user.points if current_user else 0
    goals = Goal.query.filter_by(user_id=current_user.id).order_by(Goal.created_at.desc()).all()
    return render_template('goals.html', current_user=current_user, points=points, goals=goals, is_admin_user=is_admin_user)

# Leaderboard page route
@app.route('/leaderboard')
def leaderboard():
    if 'user_email' not in session:
        flash("Please sign in to view the leaderboard and see who's leading!", 'info')
        return redirect(url_for('home'))
    current_user = get_current_user()
    points = current_user.points if current_user else 0
    users = User.query.order_by(User.points.desc()).all()
    return render_template('leaderboard.html', current_user=current_user, users=users, points=points, is_admin_user=is_admin_user)

# Chat with a specific friend route
@app.route('/chat/<int:friend_id>')
def chat_with_friend(friend_id):
    if 'user_email' not in session:
        flash("Please sign in to start chatting with your friends!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    points = current_user.points if current_user else 0
    friend = User.query.get_or_404(friend_id)
    
    if not current_user.is_friend(friend):
        flash("You can only chat with friends.", 'error')
        return redirect(url_for('chat'))
    
    return render_template("chat.html", current_user=current_user, active_friend=friend, active_group=None, points=points, is_admin_user=is_admin_user)

# Group chat route
@app.route('/group_chat/<int:group_id>')
def group_chat(group_id):
    if 'user_email' not in session:
        flash("Please sign in to join group conversations!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    points = current_user.points if current_user else 0
    group = Group.query.get_or_404(group_id)
    
    if not current_user.is_group_member(group):
        flash("You are not a member of this group.", 'error')
        return redirect(url_for('chat'))
    
    return render_template("chat.html", current_user=current_user, active_friend=None, active_group=group, points=points, is_admin_user=is_admin_user)

# Accountability partner page route
@app.route('/accountability/<friend_name>')
def accountability_partner(friend_name):
    if 'user_email' not in session:
        flash("Please sign in to view your accountability partner's details!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        flash("User session error. Please log in again.", 'error')
        return redirect(url_for('login'))
    
    partner = User.query.filter_by(name=friend_name).first()
    if not partner or not current_user.is_accountability_partner(partner):
        flash("This user is not your accountability partner.", 'error')
        return redirect(url_for('chat'))
    
    points = current_user.points if current_user else 0
    return render_template('accountability.html', current_user=current_user, partner=partner, points=points, is_admin_user=is_admin_user)

# Toggle accountability partner status
@app.route('/toggle_accountability_partner/<int:friend_id>', methods=['POST'])
def toggle_accountability_partner(friend_id):
    if 'user_email' not in session:
        flash("Please sign in to manage your accountability partners!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    friend = User.query.get(friend_id)
    if not friend:
        return jsonify({"error": "Friend not found"}), 404
    
    if not current_user.is_friend(friend):
        return jsonify({"error": "This user is not your friend"}), 400
    
    try:
        if current_user.is_accountability_partner(friend):
            current_user.remove_accountability_partner(friend)
            is_now_partner = False
        else:
            current_user.add_accountability_partner(friend)
            is_now_partner = True
        db.session.commit()
        return jsonify({"success": True, "is_now_partner": is_now_partner})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Check if a user is an accountability partner
@app.route('/is_accountability_partner/<int:friend_id>')
def is_accountability_partner(friend_id):
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    friend = User.query.get(friend_id)
    if not friend:
        return jsonify({"error": "Friend not found"}), 404
    
    return jsonify({"is_accountability_partner": current_user.is_accountability_partner(friend)})

# Get goals of an accountability partner
@app.route('/get_partner_goals/<int:partner_id>')
def get_partner_goals(partner_id):
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    partner = User.query.get(partner_id)
    if not partner:
        return jsonify({"error": "Partner not found"}), 404
    
    if not current_user.is_accountability_partner(partner):
        return jsonify({"error": "Not authorized to view this user's goals"}), 403
    
    goals = Goal.query.filter_by(user_id=partner.id).order_by(Goal.created_at.desc()).all()
    return jsonify([{
        "id": goal.id,
        "title": goal.title,
        "description": goal.description,
        "difficulty": goal.difficulty,
        "points": goal.points,
        "status": goal.status
    } for goal in goals])

# Add a new goal
@app.route('/add_goal', methods=['POST'])
def add_goal():
    if 'user_email' not in session:
        flash("Please sign in to add new goals!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    data = request.get_json()
    title = data.get('title', '').strip()
    description = data.get('description', '').strip()
    difficulty = data.get('difficulty', '').lower()

    if not title:
        return jsonify({"error": "Title is required"}), 400
    
    points_map = {'easy': 100, 'medium': 250, 'hard': 500}  # Points for each difficulty level
    if difficulty not in points_map:
        return jsonify({"error": "Invalid difficulty"}), 400

    try:
        goal = Goal(
            user_id=current_user.id,
            title=title,
            description=description,
            difficulty=difficulty,
            points=points_map[difficulty],
            status='not_started'
        )
        db.session.add(goal)
        db.session.commit()
        return jsonify({
            "success": True,
            "goal": {
                "id": goal.id,
                "title": goal.title,
                "description": goal.description,
                "difficulty": goal.difficulty,
                "points": goal.points,
                "status": goal.status
            }
        })
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Delete a goal
@app.route('/delete_goal/<int:goal_id>', methods=['POST'])
def delete_goal(goal_id):
    if 'user_email' not in session:
        flash("Please sign in to manage your goals!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    goal = Goal.query.get(goal_id)
    if not goal or goal.user_id != current_user.id:
        return jsonify({"error": "Goal not found or not authorized"}), 404
    
    try:
        db.session.delete(goal)
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Update goal status
@app.route('/update_goal_status/<int:goal_id>', methods=['POST'])
def update_goal_status(goal_id):
    if 'user_email' not in session:
        flash("Please sign in to update your goal status!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    goal = Goal.query.get(goal_id)
    if not goal or goal.user_id != current_user.id:
        return jsonify({"error": "Goal not found or not authorized"}), 404
    
    if goal.status == 'completed':
        return jsonify({"error": "Cannot change status of completed goal"}), 400
    
    data = request.get_json()
    new_status = data.get('status')
    if new_status not in ['not_started', 'in_progress']:
        return jsonify({"error": "Invalid status"}), 400
    
    try:
        goal.status = new_status
        db.session.commit()
        return jsonify({"success": True, "status": goal.status})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Complete a goal and award points
@app.route('/complete_goal/<int:goal_id>', methods=['POST'])
def complete_goal(goal_id):
    if 'user_email' not in session:
        flash("Please sign in to complete your goals!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    goal = Goal.query.get(goal_id)
    if not goal or goal.user_id != current_user.id:
        return jsonify({"error": "Goal not found or not authorized"}), 404
    
    if goal.status == 'completed':
        return jsonify({"error": "Goal already completed"}), 400
    
    try:
        goal.status = 'completed'
        current_user.points += goal.points
        db.session.commit()
        return jsonify({
            "success": True,
            "new_points": current_user.points
        })
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Get user's goals
@app.route('/get_goals')
def get_goals():
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    goals = Goal.query.filter_by(user_id=current_user.id).order_by(Goal.created_at.desc()).all()
    return jsonify([{
        "id": goal.id,
        "title": goal.title,
        "description": goal.description,
        "difficulty": goal.difficulty,
        "points": goal.points,
       }for goal in goals])

# Get user's tags
@app.route('/get_tags')
def get_tags():
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    try:
        tags = json.loads(current_user.tags)
        return jsonify({"success": True, "tags": tags})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Error retrieving tags"}), 500

# Get friend suggestions based on tag similarity
@app.route('/get_friend_suggestions')
def get_friend_suggestions():
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    try:
        all_users = User.query.filter(User.id != current_user.id).all()
        
        non_friends = []
        for user in all_users:
            if not (current_user.is_friend(user) or 
                    FriendRequest.query.filter_by(sender_id=current_user.id, receiver_id=user.id, status='pending').first() or
                    FriendRequest.query.filter_by(sender_id=user.id, receiver_id=current_user.id, status='pending').first() or
                    current_user.has_dismissed_suggestion(user)):
                non_friends.append(user)
        
        if not non_friends:
            return jsonify([])
        
        tfidf_matrix = get_tag_vectors([current_user] + non_friends)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        suggestions = []
        SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score for suggestions
        for idx, user in enumerate(non_friends):
            if similarities[idx] >= SIMILARITY_THRESHOLD:
                suggestions.append({
                    'id': user.id,
                    'name': user.name,
                    'email': user.email,
                    'similarity': float(similarities[idx])
                })
        
        suggestions.sort(key=lambda x: x['similarity'], reverse=True)
        suggestions = suggestions[:3]  # Limit to top 3 suggestions
        
        for suggestion in suggestions:
            suggestion.pop('similarity', None)
        
        return jsonify(suggestions)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Error generating friend suggestions"}), 500

# Send friend request by user ID
@app.route('/send_friend_request_by_id', methods=['POST'])
def send_friend_request_by_id():
    if 'user_email' not in session:
        flash("Please sign in to send friend requests!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    data = request.get_json()
    receiver_id = data.get('receiver_id')
    
    if not receiver_id:
        return jsonify({"error": "Receiver ID is required"}), 400
    
    receiver = User.query.get(receiver_id)
    if not receiver:
        return jsonify({"error": "User not found"}), 404
    
    try:
        if receiver.id == current_user.id:
            return jsonify({"error": "Cannot send friend request to yourself"}), 400
        if current_user.is_friend(receiver):
            return jsonify({"error": f"You are already friends with {receiver.name}"}), 400
        if FriendRequest.query.filter_by(sender_id=current_user.id, receiver_id=receiver.id, status='pending').first():
            return jsonify({"error": f"Friend request already sent to {receiver.name}"}), 400
        
        friend_request = FriendRequest(
            sender_id=current_user.id,
            receiver_id=receiver.id
        )
        db.session.add(friend_request)
        db.session.commit()
        return jsonify({"success": True, "message": f"Friend request sent to {receiver.name}"})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Remove friend suggestion
@app.route('/remove_friend_suggestion/<int:user_id>', methods=['POST'])
def remove_friend_suggestion(user_id):
    if 'user_email' not in session:
        flash("Please sign in to manage friend suggestions!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    try:
        current_user.dismiss_suggestion(user)
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Get recent messages for friends and groups
@app.route('/recent_messages')
def recent_messages():
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
        
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401

    data = []
    friends = current_user.friends.all()
    groups = current_user.groups
    friends_with_messages = []
    friends_without_messages = []
    groups_with_messages = []
    groups_without_messages = []

    for friend in friends:
        last_message = Message.query.filter(
            ((Message.sender_id == current_user.id) & (Message.receiver_id == friend.id)) |
            ((Message.sender_id == friend.id) & (Message.receiver_id == current_user.id))
        ).order_by(Message.timestamp.desc()).first()

        if last_message:
            if last_message.sender_id == current_user.id:
                msg = f"You: {last_message.content}" if last_message.content else "You: [File]"
            else:
                msg = last_message.content if last_message.content else "[File]"
            friends_with_messages.append({
                'friend_id': friend.id,
                'friend_name': friend.name,
                'message': msg,
                'profile_photo': friend.profile_photo,
                'timestamp': last_message.timestamp,
                'is_accountability_partner': current_user.is_accountability_partner(friend)
            })
        else:
            friends_without_messages.append({
                'friend_id': friend.id,
                'friend_name': friend.name,
                'message': "Start a conversation!",
                'profile_photo': friend.profile_photo,
                'timestamp': None,
                'is_accountability_partner': current_user.is_accountability_partner(friend)
            })

    for group in groups:
        last_message = GroupMessage.query.filter_by(group_id=group.id).order_by(GroupMessage.timestamp.desc()).first()
        if last_message:
            if last_message.sender_id == current_user.id:
                msg = f"You: {last_message.content}" if last_message.content else "You: [File]"
            else:
                msg = last_message.content if last_message.content else "[File]"
            groups_with_messages.append({
                'group_id': group.id,
                'group_name': group.name,
                'message': msg,
                'timestamp': last_message.timestamp,
                'is_creator': group.creator_id == current_user.id
            })
        else:
            groups_without_messages.append({
                'group_id': group.id,
                'group_name': group.name,
                'message': "Start a group conversation!",
                'timestamp': None,
                'is_creator': group.creator_id == current_user.id
            })

    friends_with_messages.sort(key=lambda x: x['timestamp'] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    groups_with_messages.sort(key=lambda x: x['timestamp'] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    data = {
        'friends': friends_without_messages + friends_with_messages,
        'groups': groups_without_messages + groups_with_messages
    }
    
    for item in data['friends']:
        item.pop('timestamp', None)
    for item in data['groups']:
        item.pop('timestamp', None)
    
    return jsonify(data)

# Create a new group
@app.route('/create_group', methods=['POST'])
def create_group():
    if 'user_email' not in session:
        flash("Please sign in to create a group!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    data = request.get_json()
    name = data.get('name', '').strip()
    members = data.get('members', [])
    
    if not name:
        return jsonify({"error": "Group name is required"}), 400
    
    try:
        group = Group(name=name, creator_id=current_user.id)
        db.session.add(group)
        db.session.flush()
        
        group.members.append(current_user)
        
        for member_id in members:
            member = User.query.get(member_id)
            if member and current_user.is_friend(member):
                group.members.append(member)
        
        db.session.commit()
        return jsonify({"success": True, "group_id": group.id})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Rename a group
@app.route('/rename_group/<int:group_id>', methods=['POST'])
def rename_group(group_id):
    if 'user_email' not in session:
        flash("Please sign in to rename a group!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    group = Group.query.get(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404
    
    if group.creator_id != current_user.id:
        return jsonify({"error": "Only the creator can rename the group"}), 403
    
    data = request.get_json()
    new_name = data.get('name', '').strip()
    
    if not new_name:
        return jsonify({"error": "New group name is required"}), 400
    
    try:
        group.name = new_name
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Leave a group
@app.route('/leave_group/<int:group_id>', methods=['POST'])
def leave_group(group_id):
    if 'user_email' not in session:
        flash("Please sign in to leave a group!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    group = Group.query.get(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404
    
    if not current_user.is_group_member(group):
        return jsonify({"error": "You are not a member of this group"}), 400
    
    try:
        group.members.remove(current_user)
        if not group.members or group.creator_id == current_user.id:
            messages = GroupMessage.query.filter_by(group_id=group.id).all()
            for message in messages:
                if message.file_path:
                    file_path = os.path.join(basedir, message.file_path[1:])
                    if os.path.exists(file_path):
                        os.remove(file_path)
                db.session.delete(message)
            db.session.delete(group)
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Send a message (private or group)
@app.route("/send_message", methods=["POST"])
def send_message():
    if 'user_email' not in session:
        flash("Please sign in to send messages!", 'info')
        return redirect(url_for('home'))
        
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401

    content = request.form.get("content", "").strip()
    recipient_id = request.form.get("recipient_id")
    group_id = request.form.get("group_id")
    files = request.files.getlist("files")

    if not content and not files:
        return jsonify({"error": "Message must contain text or file(s)"}), 400

    try:
        if recipient_id:
            recipient_id = int(recipient_id)
            recipient = User.query.get(recipient_id)
            if not recipient or not current_user.is_friend(recipient):
                return jsonify({"error": "Recipient not found or not your friend"}), 400

            if content:
                message = Message(
                    sender_id=current_user.id,
                    receiver_id=recipient_id,
                    content=content,
                    file_path=None,
                    file_size=None
                )
                db.session.add(message)

            for file in files:
                if file and file.filename != '':
                    original_filename = secure_filename(file.filename)
                    name, ext = os.path.splitext(original_filename)
                    unique_suffix = str(uuid.uuid4())
                    new_filename = f"{name}_{unique_suffix}{ext}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                    file.save(file_path)
                    public_path = f"/static/uploads/{new_filename}"

                    file_message = Message(
                        sender_id=current_user.id,
                        receiver_id=recipient_id,
                        content=None,
                        file_path=public_path,
                        file_size=os.path.getsize(file_path)
                    )
                    db.session.add(file_message)

        elif group_id:
            group_id = int(group_id)
            group = Group.query.get(group_id)
            if not group or not current_user.is_group_member(group):
                return jsonify({"error": "Group not found or you are not a member"}), 400

            if content:
                message = GroupMessage(
                    group_id=group_id,
                    sender_id=current_user.id,
                    content=content,
                    file_path=None,
                    file_size=None
                )
                db.session.add(message)

            for file in files:
                if file and file.filename != '':
                    original_filename = secure_filename(file.filename)
                    name, ext = os.path.splitext(original_filename)
                    unique_suffix = str(uuid.uuid4())
                    new_filename = f"{name}_{unique_suffix}{ext}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                    file.save(file_path)
                    public_path = f"/static/uploads/{new_filename}"

                    file_message = GroupMessage(
                        group_id=group_id,
                        sender_id=current_user.id,
                        content=None,
                        file_path=public_path,
                        file_size=os.path.getsize(file_path)
                    )
                    db.session.add(file_message)

        db.session.commit()
        return jsonify({"success": True})

    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Get private messages with a friend
@app.route("/get_messages/<int:friend_id>")
def get_messages(friend_id):
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
        
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401

    messages = Message.query.filter(
        ((Message.sender_id == current_user.id) & (Message.receiver_id == friend_id)) |
        ((Message.sender_id == friend_id) & (Message.receiver_id == current_user.id))
    ).order_by(Message.timestamp.asc()).all()

    return jsonify([{
        "id": msg.id,
        "sender_id": msg.sender_id,
        "sender_name": msg.sender.name,
        "sender_profile_photo": msg.sender.profile_photo,
        "content": msg.content,
        "file_path": msg.file_path,
        "file_size": msg.file_size,
        "timestamp": msg.timestamp.strftime("%H:%M"),
        "is_mine": msg.sender_id == current_user.id,
        "status": msg.status
    } for msg in messages])

# Get group messages
@app.route("/get_group_messages/<int:group_id>")
def get_group_messages(group_id):
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    group = Group.query.get(group_id)
    if not group or not current_user.is_group_member(group):
        return jsonify({"error": "Group not found or you are not a member"}), 400
    
    messages = GroupMessage.query.filter_by(group_id=group_id).order_by(GroupMessage.timestamp.asc()).all()
    
    return jsonify([{
        "id": msg.id,
        "sender_id": msg.sender_id,
        "sender_name": msg.sender.name,
        "sender_profile_photo": msg.sender.profile_photo,
        "content": msg.content,
        "file_path": msg.file_path,
        "file_size": msg.file_size,
        "timestamp": msg.timestamp.strftime("%H:%M"),
        "is_mine": msg.sender_id == current_user.id,
        "status": msg.status
    } for msg in messages])

# Mark private messages as seen
@app.route("/mark_seen/<int:friend_id>", methods=["POST"])
def mark_messages_seen(friend_id):
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    messages = Message.query.filter_by(
        sender_id=friend_id,
        receiver_id=current_user.id,
        status='delivered'
    ).all()
    
    for msg in messages:
        msg.status = 'seen'
    
    db.session.commit()
    
    return jsonify({"success": True})

# Mark group messages as seen
@app.route("/mark_group_seen/<int:group_id>", methods=["POST"])
def mark_group_messages_seen(group_id):
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    group = Group.query.get(group_id)
    if not group or not current_user.is_group_member(group):
        return jsonify({"error": "Group not found or you are not a member"}), 400
    
    messages = GroupMessage.query.filter_by(
        group_id=group_id,
        status='delivered'
    ).all()
    
    for msg in messages:
        if msg.sender_id != current_user.id:
            msg.status = 'seen'
    
    db.session.commit()
    
    return jsonify({"success": True})

# Delete a message
@app.route("/delete_message/<int:message_id>", methods=['POST'])
def delete_message(message_id):
    if 'user_email' not in session:
        flash("Please sign in to delete messages!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    message = Message.query.get(message_id)
    group_message = GroupMessage.query.get(message_id)
    
    if not message and not group_message:
        return jsonify({"error": "Message not found"}), 404
    
    if message and message.sender_id != current_user.id:
        return jsonify({"error": "You can only delete your own messages"}), 403
    if group_message and group_message.sender_id != current_user.id:
        return jsonify({"error": "You can only delete your own messages"}), 403
    
    try:
        target_message = message or group_message
        if target_message.file_path:
            file_path = os.path.join(basedir, target_message.file_path[1:])
            if os.path.exists(file_path):
                os.remove(file_path)
        
        db.session.delete(target_message)
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Send friend request by email
@app.route('/send_friend_request', methods=['POST'])
def send_friend_request():
    if 'user_email' not in session:
        flash("Please sign in to send friend requests!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        flash("User session error. Please log in again.", "error")
        return redirect(url_for('login'))
    
    email = request.form.get('email')
    found_user = User.query.filter_by(email=email).first()
    
    if found_user:
        if found_user.email == current_user.email:
            flash("You cannot send a friend request to yourself.", "error")
        elif current_user.is_friend(found_user):
            flash(f"You are already friends with {found_user.name}.", "info")
        elif FriendRequest.query.filter_by(sender_id=current_user.id, receiver_id=found_user.id, status='pending').first():
            flash(f"You have already sent a friend request to {found_user.name}.", "info")
        else:
            friend_request = FriendRequest(
                sender_id=current_user.id,
                receiver_id=found_user.id
            )
            db.session.add(friend_request)
            db.session.commit()
            flash(f"Friend request sent to {found_user.name}!", "success")
    else:
        flash("User not found.", "error")
    
    return redirect(url_for('chat'))

# Get incoming friend requests
@app.route('/get_friend_requests')
def get_friend_requests():
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    requests = FriendRequest.query.filter_by(receiver_id=current_user.id, status='pending').all()
    
    return jsonify([{
        'id': req.id,
        'sender_id': req.sender_id,
        'sender_name': req.sender.name,
        'sender_email': req.sender.email,
        'timestamp': req.timestamp.strftime("%Y-%m-%d %H:%M")
    } for req in requests])

# Get outgoing friend requests
@app.route('/get_outgoing_requests')
def get_outgoing_requests():
    if 'user_email' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    requests = FriendRequest.query.filter_by(sender_id=current_user.id, status='pending').all()
    
    return jsonify([{
        'id': req.id,
        'receiver_id': req.receiver_id,
        'receiver_name': req.receiver.name,
        'receiver_email': req.receiver.email,
        'timestamp': req.timestamp.strftime("%Y-%m-%d %H:%M")
    } for req in requests])

# Handle friend request (accept/reject)
@app.route('/handle_friend_request/<int:request_id>', methods=['POST'])
def handle_friend_request(request_id):
    if 'user_email' not in session:
        flash("Please sign in to manage friend requests!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    friend_request = FriendRequest.query.get(request_id)
    if not friend_request or friend_request.receiver_id != current_user.id:
        return jsonify({"error": "Friend request not found or not authorized"}), 404
    
    action = request.json.get('action')
    if action not in ['accept', 'reject']:
        return jsonify({"error": "Invalid action"}), 400
    
    try:
        if action == 'accept':
            friend_request.status = 'accepted'
            current_user.add_friend(friend_request.sender)
        else:
            friend_request.status = 'rejected'
        
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Remove a friend
@app.route('/remove_friend/<int:friend_id>', methods=['POST'])
def remove_friend(friend_id):
    if 'user_email' not in session:
        flash("Please sign in to manage your friends!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    friend = User.query.get(friend_id)
    if not friend:
        return jsonify({"error": "Friend not found"}), 404
    
    if not current_user.is_friend(friend):
        return jsonify({"error": "This user is not your friend"}), 400
    
    try:
        current_user.remove_friend(friend)
        if current_user.is_accountability_partner(friend):
            current_user.remove_accountability_partner(friend)
        db.session.commit()
        flash(f"You have unfriended {friend.name}.", "success")
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# User login
@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    if not email or not password:
        if not email:
            flash("Email cannot be blank.", 'error')
        elif not password:
            flash("Password cannot be blank.", 'error')
        return redirect(url_for("home"))

    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password, password):
        session['user_email'] = email
        session['user_name'] = user.name
        session['clear_form_state'] = True
        flash("Welcome back! You've successfully logged in.", "success")
        return redirect(url_for("home"))
    else:
        flash("Invalid email or password. Please try again.", 'error')
        return redirect(url_for("home"))

# User signup
@app.route("/signup", methods=["POST"])
def signup():
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    confirm_password = request.form.get("confirm_password")

    if not name or not email or not password or not confirm_password:
        if not name:
            flash("Name cannot be blank.", 'error')
        elif not email:
            flash("Email cannot be blank.", 'error')
        elif not password:
            flash("Password cannot be blank.", 'error')
        elif not confirm_password:
            flash("Confirm password cannot be blank.", 'error')
        return redirect(url_for("home"))

    if password != confirm_password:
        flash("Passwords do not match. Please try again.", 'error')
        return redirect(url_for("home"))

    if User.query.filter_by(email=email).first():
        flash("An account with this email already exists.", 'error')
        return redirect(url_for("home"))

    new_user = User(
        name=name,
        email=email,
        password=generate_password_hash(password),
        profile_photo=url_for('static', filename='images/default-profile.png'),
        points=0
    )
    db.session.add(new_user)
    db.session.commit()

    session['user_email'] = email
    session['user_name'] = name
    session['clear_form_state'] = True
    flash("Welcome! Your account has been created successfully.", "success")
    return redirect(url_for("chat"))

# User logout
@app.route('/logout')
def logout():
    session.pop('user_email', None)
    session.pop('user_name', None)
    if 'found_users' in session:
        session.pop('found_users', None)
    flash('You have been logged out successfully. Come back soon!', 'success')
    return redirect(url_for('home'))

# Delete user account
@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user_email' not in session:
        flash("Please sign in to delete your account!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    try:
        if current_user.profile_photo and current_user.profile_photo != url_for('static', filename='images/default-profile.png'):
            file_path = os.path.join(basedir, current_user.profile_photo[1:])
            if os.path.exists(file_path):
                os.remove(file_path)
        
        messages = Message.query.filter((Message.sender_id == current_user.id) | (Message.receiver_id == current_user.id)).all()
        for message in messages:
            if message.file_path:
                file_path = os.path.join(basedir, message.file_path[1:])
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        group_messages = GroupMessage.query.filter_by(sender_id=current_user.id).all()
        for message in group_messages:
            if message.file_path:
                file_path = os.path.join(basedir, message.file_path[1:])
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        db.session.delete(current_user)
        db.session.commit()
        
        session.pop('user_email', None)
        session.pop('user_name', None)
        if 'found_users' in session:
            session.pop('found_users', None)
        
        flash("Your account has been deleted successfully. We're sorry to see you go!", 'success')
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# User profile page
@app.route("/profile")
def profile():
    if 'user_email' not in session:
        flash("Please sign in to view your profile!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    points = current_user.points if current_user else 0
    tags = json.loads(current_user.tags) if current_user.tags else []
    return render_template("profile.html", current_user=current_user, points=points, tags=tags, is_admin_user=is_admin_user)

# Update user profile
@app.route("/update_profile", methods=["POST"])
def update_profile():
    if 'user_email' not in session:
        flash("Please sign in to update your profile!", 'info')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({"error": "User not found"}), 401
    
    if request.is_json:
        data = request.get_json()
        profile_photo = data.get("profile_photo")
        name = data.get("name", current_user.name).strip()
        tags = data.get("tags", [])
        
        try:
            current_user.name = name or current_user.name
            session['user_name'] = current_user.name
            current_user.tags = json.dumps(tags)
            
            if profile_photo == url_for('static', filename='images/default-profile.png'):
                if current_user.profile_photo and current_user.profile_photo != url_for('static', filename='images/default-profile.png'):
                    old_file_path = os.path.join(basedir, current_user.profile_photo[1:])
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)
                
                current_user.profile_photo = profile_photo
                db.session.commit()
                return jsonify({"success": True, "profile_photo": current_user.profile_photo})
        except Exception as e:
            db.session.rollback()
            traceback.print_exc()
            return jsonify({"error": "Database error"}), 500
    
    name = request.form.get("name", "").strip()
    profile_photo = request.files.get("profile_photo")
    tags = request.form.get("tags")
    tags = json.loads(tags) if tags else []

    try:
        if name:
            current_user.name = name
            session['user_name'] = name

        if profile_photo and profile_photo.filename != '':
            if not allowed_file(profile_photo.filename):
                return jsonify({"error": "Invalid file type. Only JPG/PNG allowed."}), 400
            
            profile_photo.seek(0, os.SEEK_END)
            file_size = profile_photo.tell()
            if file_size > MAX_PROFILE_PHOTO_SIZE:
                return jsonify({"error": "File too large. Max 2MB allowed."}), 400
            profile_photo.seek(0)

            original_filename = secure_filename(profile_photo.filename)
            name, ext = os.path.splitext(original_filename)
            unique_suffix = str(uuid.uuid4())
            new_filename = f"{name}_{unique_suffix}{ext}"
            file_path = os.path.join(app.config['PROFILE_PHOTO_FOLDER'], new_filename)
            profile_photo.save(file_path)
            public_path = f"/static/profile_photos/{new_filename}"

            if current_user.profile_photo and current_user.profile_photo != url_for('static', filename='images/default-profile.png'):
                old_file_path = os.path.join(basedir, current_user.profile_photo[1:])
                if os.path.exists(old_file_path):
                    os.remove(old_file_path)

            current_user.profile_photo = public_path
        
        current_user.tags = json.dumps(tags)
        db.session.commit()
        return jsonify({"success": True, "profile_photo": current_user.profile_photo})
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        return jsonify({"error": "Database error"}), 500

# Admin panel for user management
@app.route("/admin/users")
def admin_users():
    if not is_admin_user():
        flash("You are not authorized to access the admin panel.", 'error')
        return redirect(url_for('home'))
    
    current_user = get_current_user()
    points = current_user.points if current_user else 0
    users = User.query.all()
    return render_template("users.html", current_user=current_user, users=users, points=points, is_admin_user=is_admin_user)

# Delete a user (admin only)
@app.route("/admin/delete/<email>", methods=["POST"])
def delete_user(email):
    if not is_admin_user():
        flash("You are not authorized to access the admin panel.", 'error')
        return redirect(url_for('home'))
    
    user = User.query.filter_by(email=email).first()
    if not user:
        flash("User not found.", 'error')
        return redirect(url_for('admin_users'))
    
    try:
        if user.profile_photo and user.profile_photo != url_for('static', filename='images/default-profile.png'):
            file_path = os.path.join(basedir, user.profile_photo[1:])
            if os.path.exists(file_path):
                os.remove(file_path)
        
        messages = Message.query.filter((Message.sender_id == user.id) | (Message.receiver_id == user.id)).all()
        for message in messages:
            if message.file_path:
                file_path = os.path.join(basedir, message.file_path[1:])
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        group_messages = GroupMessage.query.filter_by(sender_id=user.id).all()
        for message in group_messages:
            if message.file_path:
                file_path = os.path.join(basedir, message.file_path[1:])
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        db.session.delete(user)
        db.session.commit()
        flash(f"Deleted user: {email}", 'success')
        return redirect(url_for('admin_users'))
    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        flash("Error deleting user. Please try again.", 'error')
        return redirect(url_for('admin_users'))

# Get emojis for chat
@app.route('/get-emojis')
def get_emojis():
    try:
        emoji_file_path = os.path.join(basedir, 'static', 'scripts', 'emoji.json') #json for quick loading of large data
        
        if not os.path.exists(emoji_file_path):
            return jsonify({"error": "Emoji file not found"}), 404
        
        with open(emoji_file_path, 'r', encoding='utf-8') as file:
            emojis = json.load(file)
        
        return jsonify(emojis)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Error reading emoji file"}), 500

# Initialize database and run the application
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True)  # Run in debug mode for development