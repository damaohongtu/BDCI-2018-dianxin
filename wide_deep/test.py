
def build_and_test_estimator(self, model_type):
    """Ensure that model trains and minimizes loss."""
    model = census_main.build_estimator(
        self.temp_dir, model_type,
        model_column_fn=census_dataset.build_model_columns,
        inter_op=0, intra_op=0)