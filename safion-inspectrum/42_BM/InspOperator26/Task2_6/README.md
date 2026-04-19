The original inspectrum task only allows one connected inspectrum device with a 500 K baudrate. Unfortunately, this baud rate is incompatible with the Digatron EISMETER. The orignal task can however be patched to use a baudrate of 250 K and also use different CAN IDs for communication.

You will also need a [patched version](https://git.isea.rwth-aachen.de/Personal-Projects/ESS/mso/eismeter-max-devices) of the EISmeter task to prevent the EISMeter task corrupting the inspect task.

Naming convention:
`INSP<BR>C<CI>.{EXE,MAP}`
with
- `<BR>` the baud rate identifier: `E` for 500 K, `2` for 250 K
- `<CI>` the CAN ID identifier: `T` for CAN IDs start at `0x100`, all other values represent the offset in `0x10`. E.g. for `11` the CAN IDs start at `0x100+11*0x10 = 0x1B0`
