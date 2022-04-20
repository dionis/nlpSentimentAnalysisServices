import functools

from flask import (
    Blueprint, current_app, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

#from sqlitedb import get_db
from mongodb import get_db

bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif current_app.config['DATABASE_TYPE'] == 'sqlite':
           if db.execute('SELECT id FROM user WHERE username = ?', (username,)).fetchone() is not None:
             error = f"User {username} is already registered."
        elif current_app.config['DATABASE_TYPE'] == 'mongo':
           if db['users'].find({'username',username}) is not None:
              error = f"User {username} is already registered."

        if error is None:
            # If validation succeeds, insert the new user data
            # into the database.For security, passwords should never be stored in the database
            # directly.Instead, generate_password_hash() is used
            # to securely hash the password, and that hash is stored.Since
            # this query modifies data, db.commit() needs to be called afterwards to save the changes.
           if current_app.config['DATABASE_TYPE'] == 'sqlite':
                db.execute(
                    'INSERT INTO user (username, password) VALUES (?, ?)',
                    (username, generate_password_hash(password))
                )
                db.commit()
           elif current_app.config['DATABASE_TYPE'] == 'mongo':
                 db['users'].insert_one({ 'username':username,'password':generate_password_hash(password)})
           return redirect(url_for('auth.login'))

        flash(error)

    return render_template('auth/register.html')

@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None
        if current_app.config['DATABASE_TYPE'] == 'sqlite':
            user = db.execute(
                'SELECT * FROM user WHERE username = ?', (username,)
            ).fetchone()
        elif current_app.config['DATABASE_TYPE'] == 'mongo':
            user = db['users'].find({'username':username})


        # check_password_hash() hashes the submitted password in the
        # same way as the stored hash and securely compares them.\
        # If they match, the password is valid.session is a dict that
        # stores data across requests. When validation succeeds, the user’s

        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('index'))

        flash(error)

    return render_template('auth/login.html')

@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        if current_app.config['DATABASE_TYPE'] == 'sqlite':
            g.user = get_db().execute(
                'SELECT * FROM user WHERE id = ?', (user_id,)
            ).fetchone()
        elif current_app.config['DATABASE_TYPE'] == 'mongo':
            user = get_db()['users'].find({'id': user_id})

@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))



def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view