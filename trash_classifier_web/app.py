import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

app = Flask(__name__)
model = load_model('trash_classifier_web/model/trash_classifier1.5.h5')
CLASS_NAMES = ['Cardboard', 'Clothes', 'Glass', 'Metal', 'Paper', 'Plastic', 'Shoes', 'Tidak_diketahui']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        img_file = request.files['image']

        if img_file:

            static_dir = os.path.join(app.root_path, 'static')
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)

            img_filename = img_file.filename
            img_path = os.path.join(static_dir, img_filename)
            print(f"[DEBUG] Saving uploaded file: {img_filename}")
            print(f"[DEBUG] Full path: {img_path}")
            img_file.save(img_path)
            img_url = f'/static/{img_filename}'
            print(f"[DEBUG] Image URL for template: {img_url}")

            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            import datetime
            result = model.predict(img_array)
            prediction = CLASS_NAMES[np.argmax(result)]
            confidence = f"{np.max(result) * 100:.2f}%"
            probabilities = [(CLASS_NAMES[i], float(result[0][i])) for i in range(len(CLASS_NAMES))]
            class_descriptions = {
                'Cardboard': 'Kardus dan packaging.',
                'Clothes': 'Pakaian bekas dan tekstil.',
                'Glass': 'Botol dan stoples gelas.',
                'Metal': 'Logam dan besi.',
                'Paper': 'Product kertas seperti koran dan buku.',
                'Plastic': 'Botol, wadah, and packaging plastik.',
                'Shoes': 'Sepatu bekas dan footwear.',
                'Tidak_diketahui': 'Sampah buangan yang tidak masuk ke kategori lain.'
            }
            upload_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            return render_template(
                'index.html',
                prediction=prediction,
                confidence=confidence,
                img_path=img_url,
                filename=img_filename,
                probabilities=probabilities,
                class_descriptions=class_descriptions,
                upload_time=upload_time,
            )

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
