# Background theory

`spurt` is a library of commonly used routines and workflows for transforming wrapped phase data in two or three dimensions to unwrapped phase.

## Notation

|     Symbol               |               Description          |
|--------------------------|------------------------------------|
| $\psi_{x}^{i}$           | Wrapped phase at pixel $x$ in SLC $i$ |
| $\psi_{x}^{ij}$          | Wrapped phase at pixel $x$ in interferogram $ij$                 |
| $\phi_{x}^{ij}$          | Unwrapped phase at pixel $x$ in interferogram $ij$                 |
| $\Delta \psi_{xy} ^{ij}$ |Double difference wrapped phase between pixels $x$ and $y$ in interferogram $ij$|
| $\Delta \phi_{xy}^{ij}$ | Double difference unwrapped phase between pixels $x$ and $y$ in interferogram $ij$|


Wrapped phases are assumed to be in the range $\left[ -\pi, \pi \right)$.
When working with phase linked data, the phase data in inherently referenced to a particular acquisition in the stack.

## Residue computation

Let $\lfloor \cdot \rfloor$ represent the nearest integer function. Then a residue corresponding a loop of 3 pixels $x$, $y$ and $z$ can be written in terms of wrapped phases as well as double differences depending on the unwrapping approach. We drop the superscript $ij$ for simplicity here.


### Using wrapped phases

$$
R_{xyz} = \lfloor \frac{\psi_{y} - \psi_{x}}{2\pi} \rfloor +  \lfloor \frac{\psi_{z} - \psi_{y}}{2\pi} \rfloor +  \lfloor \frac{\psi_{x} - \psi_{z}}{2\pi} \rfloor
$$

### Using double differences.

When wrapped phase differences are available

$$
R_{xyz} = \lfloor \frac{ \Delta \psi_{xy} + \Delta \psi_{yz} + \Delta \psi_{zx  } }{2 \pi} \rfloor
$$


When unwrapped phase differences are available

$$
R_{xyz} = \lfloor \frac{ \Delta \phi_{xy} + \Delta \phi_{yz} + \Delta \phi_{zx} }{2 \pi} \rfloor
$$
