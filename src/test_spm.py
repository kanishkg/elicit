import usb.core

# start a new listener thread to listen to SpaceMouse
dev = usb.core.find(idVendor=9583, idProduct=50770)
ep=dev[0].interfaces()[0].endpoints()[0]
i=dev[0].interfaces()[0].bInterfaceNumber
dev.reset()
if dev.is_kernel_driver_active(i):
    print("Detaching kernel driver ", i)
    dev.detach_kernel_driver(i)
dev.set_configuration()
eaddr = ep.bEndpointAddress
print(eaddr)
# read from the device with a timeout of 1s
r = dev.read(eaddr, 16, 10000)
print(len(r))
print(r)
