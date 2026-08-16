from anpr_app import app


if __name__ == '__main__':
    print("\nANPR System running at http://127.0.0.1:5000\n")
    app.run(debug=True, use_reloader=False)

