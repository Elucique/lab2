from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    input_values = list(map(float, request.form['Input'].split()))
    sequence = 9

    predicted = []
    input_data = []
    output_data = []

    for i in range(len(input_values) - sequence):
        input_data.append(input_values[i:i+sequence])
        output_data.append(input_values[i+sequence])

    input_data = np.array(input_data)
    output_data = np.array(output_data)

    for i in range(3):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, input_shape=(sequence,)),
            tf.keras.layers.Dense(units=1)
        ])

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
        model.fit(input_data, output_data, batch_size=512, epochs=300, verbose=0)

        predicted_value = model.predict(np.array([input_data[i]]))[0][0]
        predicted.append(predicted_value)

    asm1 = predicted[0].round(2)
    asm2 = predicted[1].round(2)
    asm3 = predicted[2].round(2)
    asm4 = ((predicted[0] + predicted[1] + predicted[2]) / 3).round(2)

    asm5 = predicted[3].round(2)
    asm6 = predicted[4].round(2)
    asm7 = predicted[5].round(2)
    asm8 = ((predicted[3] + predicted[4] + predicted[5]) / 3).round(2)

    asm9 = predicted[6].round(2)
    asm10 = (predicted[6] + 3).round(2)

    return render_template('index.html', asm1=asm1, asm2=asm2, asm3=asm3, asm4=asm4, asm5=asm5,
                           asm6=asm6, asm7=asm7, asm8=asm8, asm9=asm9, asm10=asm10)

if __name__ == '__main__':
    app.run()
