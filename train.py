from data_generation2 import load_dataset_soccernet_m, designate_batches, DataGenerator
from modeling import JerseyNumberRecognition

# json_file_path = 'C:/Users/HassenBELHASSEN/PycharmProjects/sn-jersey-number-annotation/train_annotations.json'
# base_path = 'C:/Users/HassenBELHASSEN/AppData/Local/Programs/Python/Python311/Lib/site-packages/SoccerNet'
# additional_images_dir = 'C:/Users/HassenBELHASSEN/Desktop/only_m1'

json_file_path = 'C:/Users/Administrator/PycharmProjects/jersey_number_recognition/jersey_number/train_annotations.json'
base_path = 'D:/soccernet_jersey/'
additional_images_dir = 'D:/soccernet_jersey/'

train_dataset, valid_dataset = load_dataset_soccernet_m(json_file_path, base_path)
print(f"train", len(train_dataset))
print(f"valid", len(valid_dataset))

print(train_dataset[0])

batch_size = 64
train_batches, num_train_batches = designate_batches(train_dataset, batch_size=batch_size)
valid_batches, num_valid_batches = designate_batches(valid_dataset, batch_size=batch_size)

print(num_train_batches)
print(num_valid_batches)

train_gen = DataGenerator(train_batches, batch_size=batch_size, image_size=(64, 64))
print(train_gen.length)
valid_gen = DataGenerator(valid_batches, batch_size=batch_size, image_size=(64, 64))
print(valid_gen.length)

jnr = JerseyNumberRecognition()
jnr.compile_model()
history = jnr.train_model_generator(train_gen=train_gen, valid_gen=valid_gen
                                    , training_steps=train_gen.length, validation_steps=valid_gen.length,
                                    checkpoint_filepath='models/model_lost_3.keras',
                                    epochs=10, batch_size=batch_size)
