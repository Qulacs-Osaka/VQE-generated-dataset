OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.1928416289745973) q[0];
rz(-1.541861570788984) q[0];
ry(-0.08588687219841326) q[1];
rz(-2.5487922223925366) q[1];
ry(2.623346156662643) q[2];
rz(0.10020303360057604) q[2];
ry(2.600944296958659) q[3];
rz(-2.4547258129370624) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.3653199949795949) q[0];
rz(-0.30053104029882055) q[0];
ry(-2.6808140500645696) q[1];
rz(-1.1010803646228478) q[1];
ry(0.01656160636393099) q[2];
rz(0.8841470762586803) q[2];
ry(0.055294066882454335) q[3];
rz(-2.2723352385366526) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.75454816468408) q[0];
rz(0.14670705565070905) q[0];
ry(0.9337988455987007) q[1];
rz(1.2706340011070747) q[1];
ry(0.013143626225582558) q[2];
rz(2.4064532247948622) q[2];
ry(-2.7716452087821333) q[3];
rz(1.22506361694494) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.205802215731282) q[0];
rz(0.3289517304884991) q[0];
ry(0.0547322152713523) q[1];
rz(-1.1600243604717786) q[1];
ry(3.0364699674490363) q[2];
rz(-2.4017297511309543) q[2];
ry(-0.24241344928649236) q[3];
rz(2.882015127106384) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.7730465870723666) q[0];
rz(-1.2107491991515071) q[0];
ry(-2.2374365257671993) q[1];
rz(-2.3174586530523262) q[1];
ry(1.5757462204965256) q[2];
rz(2.8094006909832347) q[2];
ry(-1.6591222316253305) q[3];
rz(0.043855654841126857) q[3];