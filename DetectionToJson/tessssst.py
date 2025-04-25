from pypylon import pylon
devices = pylon.TlFactory.GetInstance().EnumerateDevices()
print("Found devices:", devices)
