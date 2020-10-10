from translation_xx_to_yy.tensorflow._1_setup_input_pipeline.script import SetupInputPipeline
from translation_xx_to_yy.tensorflow._3_masking.script import Masking
from translation_xx_to_yy.tensorflow._11_create_the_transformer.script import Transformer
from translation_xx_to_yy.tensorflow._12_set_hyper_paramters.script import *
from translation_xx_to_yy.tensorflow._13_optimizer.script import CustomSchedule
from translation_xx_to_yy.tensorflow._14_loss_and_metrics.script import loss_function, train_loss, train_accuracy
import tensorflow as tf
import time

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = Masking.create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = Masking.create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = Masking.create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = Masking.create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


if __name__ == '__main__':
    EPOCHS = 20
    # load the dataset
    pipeline = SetupInputPipeline()
    pipeline.init_ted_hrlr()
    pipeline.init_subwords_tokenizer()
    train_dataset = pipeline.preproc_train_ex()

    # init a transformer
    transformer = Transformer(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF,
                              INPUT_VOCAB_SIZE, TARGET_VOCAB_SIZE,
                              pe_input=INPUT_VOCAB_SIZE,
                              pe_target=TARGET_VOCAB_SIZE,
                              rate=DROPOUT_RATE)

    checkpoint_path = "./checkpoints/train"
    # create a custom schedule
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # train
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            # save checkpoint every 5 epoch.
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
