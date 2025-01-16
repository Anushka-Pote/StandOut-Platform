from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User, Message

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
app.config['SECRET_KEY'] = 'your_secret_key'

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)

# User Loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password_hash == password:  # Replace with hashed password check
            login_user(user)
            return redirect(url_for('chat'))
        flash('Invalid credentials')
    return render_template('chat.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if request.method == 'POST':
        new_message = request.form['message']
        db.session.add(Message(user_id=current_user.user_id, message_content=new_message))
        db.session.commit()

    messages = Message.query.filter_by(user_id=current_user.user_id, is_deleted=False).all()
    return render_template('chat.html', messages=messages)

@app.route('/edit_message/<int:message_id>', methods=['GET', 'POST'])
@login_required
def edit_message(message_id):
    message = Message.query.get(message_id)
    if message.user_id != current_user.user_id:
        return redirect(url_for('chat'))  # Prevent editing someone else's message
    if request.method == 'POST':
        message.message_content = request.form['message']
        db.session.commit()
        return redirect(url_for('chat'))
    return render_template('edit_message.html', message=message)

@app.route('/delete_message/<int:message_id>', methods=['POST'])
@login_required
def delete_message(message_id):
    message = Message.query.get(message_id)
    if message.user_id != current_user.user_id and current_user.role != 'admin':
        return redirect(url_for('chat'))  # Prevent deleting someone else's message unless admin
    message.is_deleted = True
    db.session.commit()
    return redirect(url_for('chat'))

@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin':
        return redirect(url_for('chat'))  # Only admins can access this
    messages = Message.query.all()
    return render_template('admin_panel.html', messages=messages)

if __name__ == '__main__':
    app.run(debug=True)
