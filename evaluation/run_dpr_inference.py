from methods.ir.dense.dpr.models.dpr_sentence_transformers_inference import DprSentSearch

from data.datastructures.hyperparameters.dpr import DprHyperParams


if __name__ == "__main__":

    config_instance = DprHyperParams(query_encoder_path="facebook-dpr-question_encoder-single-nq-base",
                                     document_encoder_path="facebook-dpr-ctx_encoder-single-nq-base",
                                     ann_search="annoy_search")
   # config = config_instance.get_all_params()

    dpr_sent_search = DprSentSearch(config_instance)
    _ = dpr_sent_search.get_ann_algo(768, 100, "euclidean")
    dpr_sent_search.create_index(
        "/raid_data-lv/venktesh/AmbigQA/codes/data/data/wikipedia_split/psgs_w100.tsv.gz", 100)
    indices = dpr_sent_search.retrieve(
        ["When did the Simpsons first air on television as an animated short on the Tracey Ullman Show?"], 100)
    print("indices",indices,len(indices))