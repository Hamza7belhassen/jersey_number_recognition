import tensorflow as tf


def train_model_generator(model,train_gen, valid_gen, training_steps, checkpoint_filepath, validation_steps,
                          epochs=50, batch_size=32, verbose=1):

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    callbacks = [model_checkpoint_callback]

    history = model.fit(
        train_gen,  # Training generator
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        steps_per_epoch=training_steps,
        validation_data=valid_gen,  # Validation generator
        validation_steps=validation_steps,  # Number of validation steps
        callbacks=callbacks
    )
    return history
