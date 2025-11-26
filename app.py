from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index(): #Rota da página principal.
    return render_template("index.html")

@app.route('/mapa')
def mapa(): #Rota da página do mapa interativo.
    return render_template("mapa.html")

if __name__ == '__main__':
    app.run(debug=True)