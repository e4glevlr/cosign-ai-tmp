mt5_path = "google/mt5-base"

# label paths
train_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train",
                    "VSL": "./vn_sentence_data/labels.train",  # Vietnamese Sign Language
                    }

dev_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-2000.dev",
                    "VSL": "./vn_sentence_data/labels.dev",
                    }

test_label_paths = {
                    "CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-2000.test",
                    "VSL": "./vn_sentence_data/labels.test",
                    }


# video paths
rgb_dirs = {
            "CSL_News": './dataset/CSL_News/rgb_format',
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format",
            "VSL": "./vn_sentence_data/video",
            }

# pose paths (RTMPose extracted)
pose_dirs = {
            "CSL_News": './dataset/CSL_News/pose_format',
            "CSL_Daily": './dataset/CSL_Daily/pose_format',
            "WLASL": "./dataset/WLASL/pose_format",
            "VSL": "./vn_sentence_data/pose",
            }

# Apple Vision pose paths (for fine-tuning)
pose_dirs_apple = {
            "CSL_News": './dataset/CSL_News/pose_format_apple',
            "CSL_Daily": './dataset/CSL_Daily/pose_format_apple',
            "WLASL": "./dataset/WLASL/pose_format_apple",
            "VSL": "./vn_sentence_data/pose_apple",
            }