from argparse import ArgumentParser

from augment_data import augment_data
from predict_species import predict_species
from train_model import train_model


def build_parser() -> ArgumentParser:
	parser = ArgumentParser(description="Bird CNN command-line interface")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--augment", action="store_true", help="Prepare or expand the dataset")
	group.add_argument("--train", action="store_true", help="Train the model")
	group.add_argument("--predict", action="store_true", help="Predict a bird species from an image")
	parser.add_argument("--image", help="Path to the image to classify")
	return parser


def main() -> None:
	args = build_parser().parse_args()

	if args.augment:
		augment_data()
	elif args.train:
		train_model()
	elif args.predict:
		predict_species(args.image)


if __name__ == "__main__":
	main()
