import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./fv101_model_trained.hdf5')
	return model


def predict_class(image, model):

	#image = tf.cast(image, tf.float32)
	#image = tf.image.resize(image, [224,224])
	image = tf.convert_to_tensor(image, dtype=tf.float32)

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
#This is for the instructions home page
st.title('FOODVISION101 WEB APP')
Main_image = st.image('https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/img/food-101.jpg',caption='Source: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ ')
#This is load the markdown page for the entire home page
def get_file_content_as_string(path_1):
    path = os.path.dirname(__file__)
    my_file = path + path_1
    with open(my_file,'r') as f:
        instructions=f.read()
    return instructions
readme_text=st.markdown(get_file_content_as_string('Instructions.md'), unsafe_allow_html=True)
file = st.file_uploader("Upload an image of food", type=["jpg", "png"])
if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(file)
	test_image = test_image.resize((224,224))

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito','bruschetta','caesar_salad',
	'cannoli','caprese_salad','carrot_cake','ceviche','cheesecake','cheese_plate','chicken_curry','chicken_quesadilla','chicken_wings','chocolate_cake','chocolate_mousse','churros',
    'clam_chowder','club_sandwich','crab_cakes','creme_brulee','croque_madame','cup_cakes','deviled_eggs','donuts','dumplings','edamame','eggs_benedict','escargots','falafel',
    'filet_mignon','fish_and_chips','foie_gras','french_fries','french_onion_soup','french_toast','fried_calamari','fried_rice','frozen_yogurt','garlic_bread','gnocchi',
    'greek_salad','grilled_cheese_sandwich','grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup','hot_dog','huevos_rancheros','hummus','ice_cream','lasagna',
    'lobster_bisque','lobster_roll_sandwich','macaroni_and_cheese','macarons','miso_soup','mussels','nachos','omelette','onion_rings','oysters','pad_thai','paella','pancakes',
    'panna_cotta','peking_duck','pho','pizza','pork_chop','poutine','prime_rib','pulled_pork_sandwich','ramen','ravioli','red_velvet_cake','risotto','samosa','sashimi',
    'scallops','seaweed_salad','shrimp_and_grits','spaghetti_bolognese','spaghetti_carbonara','spring_rolls','steak','strawberry_shortcake','sushi','tacos','takoyaki','tiramisu',
    'tuna_tartare','waffles']

	result = class_names[np.argmax(pred)]
	op = np.argmax(pred)

	output = 'The image is a ' + str(op)

	slot.text('Done')

	st.success(output)


#function to show the developer information
st.sidebar.markdown("# A B O U T")
st.sidebar.image("https://i.imgur.com/DKqiF7I.png",width=180)
st.sidebar.markdown("## Viraj Jadhav")
st.sidebar.markdown('* ####  Connect via [LinkedIn](http://linkedin.com/in/viraj-jadhav-7b3970219)')
st.sidebar.markdown('* ####  Connect via [Github](https://github.com/viraj-19)')
st.sidebar.markdown('* ####  19jadhavviraj@gmail.com')
