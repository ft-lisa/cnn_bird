import random
import shutil
from pathlib import Path
from collections.abc import Iterable
import random


from PIL import Image, ImageEnhance, ImageFilter, ImageOps


class DatasetAugmenter:
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        results_dir: Path | None = None,
        output_dir: Path | None = None,
        target_count: int = 5000,
    ) -> None:
        self.results_dir = results_dir or Path(__file__).resolve().parents[1] / "results"
        self.output_dir = output_dir or Path(__file__).resolve().parents[1] / "augmented_results"
        self.target_count = target_count
        self.random = random.Random()

    def get_species_image_directories(self) -> list[Path]:
        if not self.results_dir.exists():
            return []

        return sorted(
            path
            for path in self.results_dir.iterdir()
            if path.is_dir() and path.name.endswith("_images")
        )

    def list_image_files(self, image_directory: Path) -> list[Path]:
        return sorted(
            path
            for path in image_directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.allowed_extensions
        )

    def count_photos(self, image_directory: Path) -> int:
        return len(self.list_image_files(image_directory))

    def augment_all_species(self) -> None:
        train_dir = self.output_dir / "train"
        stats_dir = self.output_dir / "stats"

        image_directories = self.get_species_image_directories()  # scan self.results_dir (dataset original)

        if not image_directories:
            print(f"No species image folders found in {self.results_dir}")
            return

        self.split_dataset(image_directories, train_dir, stats_dir, 42)

        # Augmentation uniquement sur train — stats reste intact pour une évaluation fiable
        for species_directory in sorted(train_dir.iterdir()):
            if not species_directory.is_dir():
                continue
            species_name = species_directory.name.removesuffix("_images")
            created_photos = self.augment_species_folder(species_directory, species_name)
            final_count = self.count_photos(species_directory)
            print(f"{species_name}: {final_count} photos in train (+{created_photos} created)")


    def augment_species_folder(self, species_directory: Path, species_name: str) -> int:
        source_images = self.list_image_files(species_directory)

        if not source_images:
            return 0

        current_count = self.count_photos(species_directory)
        if current_count >= self.target_count:
            return 0

        created_count = 0
        source_index = 0

        while current_count + created_count < self.target_count:
            source_image = source_images[source_index % len(source_images)]
            source_index += 1

            with Image.open(source_image) as original_image:
                augmented_image = self.create_augmented_image(original_image)
                output_path = self.build_output_path(
                    species_directory, source_image, current_count + created_count + 1
                )
                self.save_image(augmented_image, output_path)

            created_count += 1

        return created_count
    

    def split_dataset(
    self,
    species_dirs: Iterable[Path],
    train_dir: Path,
    stats_dir: Path,
    seed: int,
    ) -> dict[str, dict[str, int]]:
        random_generator = random.Random(seed)
        summary: dict[str, dict[str, int]] = {}

        for species_dir in species_dirs:
            species_name = species_dir.name.removesuffix("_images")
            image_paths = sorted(
                path
                for path in species_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in self.allowed_extensions
            )

            if not image_paths:
                summary[species_name] = {"train": 0, "stats": 0}
                continue

            shuffled_paths = image_paths[:]
            random_generator.shuffle(shuffled_paths)

            stats_count = max(1, round(len(shuffled_paths) * 0.1)) if len(shuffled_paths) > 1 else 0
            stats_count = min(stats_count, len(shuffled_paths) - 1) if len(shuffled_paths) > 1 else 0
            stats_paths = shuffled_paths[:stats_count]
            train_paths = shuffled_paths[stats_count:]

            self.copy_images(train_paths, train_dir / species_dir.name)
            self.copy_images(stats_paths, stats_dir / species_dir.name)

            summary[species_name] = {"train": len(train_paths), "stats": len(stats_paths)}

        return summary

    
    def copy_images(self, source_paths: list[Path], destination_dir: Path) -> None:
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source_path in source_paths:
            shutil.copy2(source_path, destination_dir / source_path.name)


    def copy_original_images(self, source_images: list[Path], output_directory: Path) -> None:
        for source_image in source_images:
            destination_path = output_directory / source_image.name
            if destination_path.exists():
                continue

            shutil.copy2(source_image, destination_path)

    def get_output_species_directory(self, species_name: str) -> Path:
        return self.output_dir / f"{species_name}_images"

    def build_output_path(self, output_directory: Path, source_image: Path, sequence_number: int) -> Path:
        return output_directory / f"augmented_{sequence_number:04d}_{source_image.stem}{source_image.suffix.lower()}"

    def save_image(self, image: Image.Image, output_path: Path) -> None:
        image_to_save = image
        if output_path.suffix.lower() in {".jpg", ".jpeg"} and image.mode not in {"RGB", "L"}:
            image_to_save = image.convert("RGB")

        image_to_save.save(output_path)

    def create_augmented_image(self, image: Image.Image) -> Image.Image:
        transformed_image = image.copy().convert("RGB")

        transformation_steps = [
            self.apply_rotation,
            self.apply_zoom,
            self.apply_blur,
            self.apply_horizontal_flip,
            self.apply_brightness,
            self.apply_contrast,
        ]

        self.random.shuffle(transformation_steps)
        for transform in transformation_steps[: self.random.randint(2, 4)]:
            transformed_image = transform(transformed_image)

        return transformed_image

    def apply_rotation(self, image: Image.Image) -> Image.Image:
        angle = self.random.uniform(-30, 30)
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

    def apply_zoom(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        zoom_factor = self.random.uniform(0.8, 1.0)
        crop_width = max(1, int(width * zoom_factor))
        crop_height = max(1, int(height * zoom_factor))

        left = self.random.randint(0, max(0, width - crop_width))
        top = self.random.randint(0, max(0, height - crop_height))

        cropped_image = image.crop((left, top, left + crop_width, top + crop_height))
        return ImageOps.fit(cropped_image, (width, height), method=Image.Resampling.LANCZOS)

    def apply_blur(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=self.random.uniform(0.2, 1.5)))

    def apply_horizontal_flip(self, image: Image.Image) -> Image.Image:
        return ImageOps.mirror(image)

    def apply_brightness(self, image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(self.random.uniform(0.75, 1.25))

    def apply_contrast(self, image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(self.random.uniform(0.75, 1.25))


def augment_data() -> None:
    augmenter = DatasetAugmenter()
    augmenter.augment_all_species()