# Habit planes between austenite and a single variant of martensite in NiTi under stress
Model predicting the habit planes between elastically distorted cubic austenite and monoclinic martensite in NiTi shape memory alloys.

The use of the model is explained in the jupyter notebook in the directory notebook.

To evaluate the effect of elastic loading on the strain compatibility at the habit plane interface, we utilized standard PTMC theory, where, we consider sets of three independent basal lattice vectors of the elastically deformed lattices of austenite and martensite. The elastically distorted lattice vectors were utilized to compute the transformation deformation gradients, which are then inserted into the compatibility equation for the purpose of verifying the presence of a habit plane. For a thorough description of how the transformation deformation gradients between elastically deformed lattices of austenite and martensite were assessed, the reader is asked to read Appendix A. The gradients are completely determined by the lattice parameters of the B2 cubic austenite, the B19’ monoclinic martensite, the orientation relationship between the two lattices, their elastic constants, and the applied stress.
As stated in the introduction, we assume that the elastic strain has no impact on the nature of the MT in NiTi, despite that it affects the lattice symmetries. Therefore, we consider the same orientation relationship and lattice correspondence as observed under the stress-free condition. However, we account for the change in transformation deformation gradients by considering elastically distorted basal lattice vectors of austenite and martensite.

More details can be found in the related publication:
HELLER, Luděk, ŠITTNER, Petr, On the habit planes between elastically distorted austenite and martensite in NiTi, Acta Materialia. 2024, 119828. ISSN 1359-6454. E-ISSN 1873-2453, doi:10.1016/j.actamat.2024.119828.
