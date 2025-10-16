from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow.keras.models import load_model


test_df=pd.read_csv("final_test_with_category.csv")

@tf.keras.saving.register_keras_serializable()
def smape_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = tf.abs(y_true - y_pred)
    denominator = tf.maximum((tf.abs(y_true) + tf.abs(y_pred)) / 2.0, 1e-3)
    smape = numerator / denominator
    smape = tf.where(tf.math.is_finite(smape), smape, tf.zeros_like(smape))
    return tf.reduce_mean(smape)


best_model = load_model("mode.h5", custom_objects={'smape_loss': smape_loss})


y_pred = best_model.predict(X_test)
print(np.isnan(y_pred).sum())



test_out = pd.DataFrame(
    {"sample_id": test_df["sample_id"].values, "price": y_pred.flatten()})

test_out.isnull().sum()
test_out.to_csv("test_out.csv", index=False)
