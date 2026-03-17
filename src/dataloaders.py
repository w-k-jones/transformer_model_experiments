
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

class GeoDataset(Dataset):
    def __init__(self, files, satellite_angles=False):
        super().__init__()
        self.files = files
        self.satellite_angles = satellite_angles
    
    def __len__(self):
        return len(self.files)

    def load_image(self, idx):
        with xr.open_datatree(self.files[idx]) as dt:
            image_input = dt.geo_patch.data.fillna(0)
            image_input = image_input[:3] / 100 # 3 channel input
            # image_input[6:] = image_input[6:] / 140 - 180
            image_input = np.clip(image_input, 0, 1).values.astype(np.float32)

            ds = dt.geo_patch
            if "latitude" in ds.data_vars and "longitude" in ds.data_vars:
                lat_offset = (
                    (ds.latitude.encoding["scale_factor"] + ds.latitude.encoding["add_offset"])*2
                    if ds.latitude.encoding["add_offset"] > 0
                    else 0
                )
                lon_offset = (
                    (ds.longitude.encoding["scale_factor"] + ds.longitude.encoding["add_offset"])*2
                    if ds.longitude.encoding["add_offset"] > 0
                    else 0
                )
                latitudes = (ds.latitude - lat_offset).fillna(ds.latitude.encoding["add_offset"])
                longitudes = (ds.longitude - lon_offset).fillna(ds.longitude.encoding["add_offset"])
                time_epoch = np.full(dt.geo_patch.longitude.shape, (dt.geo_patch.t.values - np.datetime64("1970-01-01T00:00:00"))/ np.timedelta64(1, "D"), dtype=np.float32)
                time_of_day = np.full(dt.geo_patch.longitude.shape, (dt.geo_patch.t.values - dt.geo_patch.t.values.astype("datetime64[D]")) / np.timedelta64(1, "D"), dtype=np.float32)

            coords = np.stack([
                longitudes, 
                latitudes, 
                # time_of_day,
                # time_epoch,
            ], 0).astype(np.float32)

            if self.satellite_angles:
                sat_azi = ds.sat_angle[1]
                sat_zen = ds.sat_angle[0]
                coords = np.concatenate([
                    coords, 
                    np.stack([
                        sat_azi,
                        sat_zen,
                    ], 0).astype(np.float32)
                ], 0)
            
        return image_input, coords
    
    def __getitem__(self, idx):
        return self.load_image(idx)

class GeoDataloader(LightningDataModule):
    def __init__(
        self, 
        batch_size, 
        files, 
        num_workers=0, 
        satellite_angles=False, 
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Get list of files for each split
        self.train_dataset = GeoDataset(files[:int(len(files)*0.7)], satellite_angles=satellite_angles)
        self.val_dataset = GeoDataset(files[int(len(files)*0.75):int(len(files)*0.85)], satellite_angles=satellite_angles)
        self.test_dataset = GeoDataset(files[int(len(files)*0.9):], satellite_angles=satellite_angles)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=False, 
            num_workers=self.num_workers, 
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=False, 
            num_workers=self.num_workers, 
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=False, 
            num_workers=self.num_workers, 
        )

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
class GeoCloudsatDataset(Dataset):
    def __init__(self, files, satellite_angles=False):
        super().__init__()
        self.files = files
        self.satellite_angles = satellite_angles
    
    def __len__(self):
        return len(self.files)

    def load_image_and_geoloc(self, idx):
        with xr.open_datatree(self.files[idx]) as dt:
            image_input = dt.geo_patch.data.fillna(0)
            image_input[:6] = image_input[:6] / 100
            image_input[6:] = (image_input[6:] - 180) / 140 
            image_input = np.clip(image_input, 0, 1).values.astype(np.float32)

            # Fix latitude/longitude encoding:
            ds = dt.geo_patch
            if "latitude" in ds.data_vars and "longitude" in ds.data_vars:
                lat_offset = (
                    (ds.latitude.encoding["scale_factor"] + ds.latitude.encoding["add_offset"])*2
                    if ds.latitude.encoding["add_offset"] > 0
                    else 0
                )
                lon_offset = (
                    (ds.longitude.encoding["scale_factor"] + ds.longitude.encoding["add_offset"])*2
                    if ds.longitude.encoding["add_offset"] > 0
                    else 0
                )
                latitudes = (ds.latitude - lat_offset).fillna(ds.latitude.encoding["add_offset"])
                longitudes = (ds.longitude - lon_offset).fillna(ds.longitude.encoding["add_offset"])
            time_epoch = np.full(dt.geo_patch.longitude.shape, (dt.geo_patch.t.values - np.datetime64("1970-01-01T00:00:00"))/ np.timedelta64(1, "D"), dtype=np.float32)
            time_of_day = np.full(dt.geo_patch.longitude.shape, (dt.geo_patch.t.values - dt.geo_patch.t.values.astype("datetime64[D]")) / np.timedelta64(1, "D"), dtype=np.float32)
            
            coords = np.stack([
                longitudes, 
                latitudes, 
                # time_of_day,
                # time_epoch,
            ], 0).astype(np.float32)

            if self.satellite_angles:
                sat_azi = ds.sat_angle[1]
                sat_zen = ds.sat_angle[0]
                coords = np.concatenate([
                    coords, 
                    np.stack([
                        sat_azi,
                        sat_zen,
                    ], 0).astype(np.float32)
                ], 0)

            # Sample centred 256 columns
            cloudsat_offset = (dt.cloudsat_unaligned.Nray.size-256)//2
            cloudsat_unaligned = dt.cloudsat_unaligned.isel(Nray=slice(cloudsat_offset, cloudsat_offset+256))

            # Randomly sample 256 columns to target
            # cloudsat_inds = np.random.choice(np.arange(dt.cloudsat_unaligned.Nray.size), 256)
            # cloudsat_unaligned = dt.cloudsat_unaligned.isel(Nray=cloudsat_inds)

            target = cloudsat_unaligned.Radar_Reflectivity.fillna(-30).values.astype(np.float32)

            time_epoch = ((cloudsat_unaligned.Profile_time.values - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "D")).astype(np.float32)
            time_of_day = ((cloudsat_unaligned.Profile_time.values - cloudsat_unaligned.Profile_time.values.astype("datetime64[D]")) / np.timedelta64(1, "D")).astype(np.float32)
            
            geoloc_output = np.stack([
                cloudsat_unaligned.Longitude.values, 
                cloudsat_unaligned.Latitude.values, 
                # time_of_day,
                # time_epoch,
            ], 0).astype(np.float32)

            if self.satellite_angles:
                geoloc_output = np.concatenate([
                    geoloc_output,
                    np.zeros_like(geoloc_output[:2])
                ], 0)

            
        return image_input, coords, target, geoloc_output
    
    def __getitem__(self, idx):
        return self.load_image_and_geoloc(idx)

class GeoCloudsatDataloader(LightningDataModule):
    def __init__(
        self, 
        batch_size, 
        files, 
        satellite_angles=False,
        **dataloader_kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs

        # Get list of files for each split
        self.train_dataset = GeoCloudsatDataset(files[:int(len(files)*0.7)], satellite_angles=satellite_angles)
        self.val_dataset = GeoCloudsatDataset(files[int(len(files)*0.75):int(len(files)*0.85)], satellite_angles=satellite_angles)
        self.test_dataset = GeoCloudsatDataset(files[int(len(files)*0.9):], satellite_angles=satellite_angles)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            **self.dataloader_kwargs, 
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            **self.dataloader_kwargs, 
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            **self.dataloader_kwargs, 
        )
