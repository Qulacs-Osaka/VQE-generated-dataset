OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.2818982960955079) q[0];
rz(2.6805215273977754) q[0];
ry(2.646533738247753) q[1];
rz(0.6995576351946329) q[1];
ry(1.858162875919221) q[2];
rz(1.6484416485299698) q[2];
ry(0.9140318194029087) q[3];
rz(-0.6671877861506639) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.188709564569023) q[0];
rz(0.705373851218468) q[0];
ry(2.4227907274854874) q[1];
rz(-1.381115099079147) q[1];
ry(0.728969918056282) q[2];
rz(2.192558774116174) q[2];
ry(2.949243833353784) q[3];
rz(-2.826067285019429) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.218011869385217) q[0];
rz(-3.1112448561863006) q[0];
ry(-2.0642815769880896) q[1];
rz(-1.1485864896872708) q[1];
ry(-1.4860343046138882) q[2];
rz(-0.2990010277090534) q[2];
ry(0.8534520152847458) q[3];
rz(0.28290242412485256) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.5126358634576862) q[0];
rz(-1.1018375059986916) q[0];
ry(1.7770962956292058) q[1];
rz(3.123026323092693) q[1];
ry(0.36542713506725044) q[2];
rz(-0.9792838551985943) q[2];
ry(0.055306640391164494) q[3];
rz(0.3078246946604691) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.1095139406536636) q[0];
rz(2.2685227709642186) q[0];
ry(-2.0022429062763005) q[1];
rz(-2.908671515491127) q[1];
ry(0.3533779226762622) q[2];
rz(2.5674956143426972) q[2];
ry(-2.462061663585744) q[3];
rz(-2.7769781776089104) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.664037703147532) q[0];
rz(-1.6919869780383021) q[0];
ry(0.997191127579778) q[1];
rz(-0.5740746901472803) q[1];
ry(-3.0973719329330174) q[2];
rz(-2.0448062857046616) q[2];
ry(-0.6033964081232566) q[3];
rz(1.6397540583484629) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.1559435530160345) q[0];
rz(3.0194228149397793) q[0];
ry(0.30755024522592134) q[1];
rz(2.8607989296459464) q[1];
ry(0.8550308685059308) q[2];
rz(0.655024497462204) q[2];
ry(0.3724863127434421) q[3];
rz(-2.2377502374944953) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3354923180417018) q[0];
rz(0.08575947946082305) q[0];
ry(1.8205935522906942) q[1];
rz(2.955385651507077) q[1];
ry(0.4973331412330637) q[2];
rz(2.2599145175589634) q[2];
ry(-1.2058117118140466) q[3];
rz(3.0695861082767917) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.867989643440724) q[0];
rz(-0.5935081691638419) q[0];
ry(-2.8244396271776617) q[1];
rz(0.9875251132474379) q[1];
ry(1.3153490114276378) q[2];
rz(0.8805919921202977) q[2];
ry(2.5381489718733987) q[3];
rz(-1.9764878050065486) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.451747964553817) q[0];
rz(-2.371940705375646) q[0];
ry(0.9330178847768001) q[1];
rz(1.1407502962307867) q[1];
ry(-0.5109077524193347) q[2];
rz(0.15812561544491383) q[2];
ry(1.0922991909363944) q[3];
rz(2.5853221167376548) q[3];