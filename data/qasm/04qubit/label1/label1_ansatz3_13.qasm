OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.1700479996471951) q[0];
rz(0.03733650229826534) q[0];
ry(-1.2290018704056687) q[1];
rz(-0.6268550274638846) q[1];
ry(2.152249049385888) q[2];
rz(-1.3563604455134077) q[2];
ry(1.9956530180443481) q[3];
rz(-1.4914601770324376) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.04229137963921771) q[0];
rz(0.2980159563621988) q[0];
ry(2.98639884486349) q[1];
rz(2.968893693429695) q[1];
ry(-2.90443775295973) q[2];
rz(-1.0493758070385582) q[2];
ry(-1.4817539177458983) q[3];
rz(-0.9297134493473357) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.4105362339426384) q[0];
rz(-2.5028603345100158) q[0];
ry(0.8521674537198473) q[1];
rz(-2.0045628839145477) q[1];
ry(0.1083837048930576) q[2];
rz(1.8393136475148135) q[2];
ry(1.619768144010413) q[3];
rz(-2.3261026150352153) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.291785956011533) q[0];
rz(-2.1762605970557627) q[0];
ry(0.7601067840704667) q[1];
rz(-2.3281594335985867) q[1];
ry(-1.5255784417963687) q[2];
rz(-0.710341294346128) q[2];
ry(-0.3940587717494748) q[3];
rz(2.659780221676015) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.943023681622841) q[0];
rz(1.1678948791490975) q[0];
ry(-2.597059132824371) q[1];
rz(-0.10189747653614853) q[1];
ry(1.545532393833779) q[2];
rz(1.451262395932437) q[2];
ry(-0.8197103231983895) q[3];
rz(-0.3193313220375815) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.598756935522072) q[0];
rz(0.1744175827368851) q[0];
ry(2.161830109684224) q[1];
rz(0.09788787403969312) q[1];
ry(0.7761668041713552) q[2];
rz(1.4227744804104736) q[2];
ry(3.0949261393628347) q[3];
rz(2.544261854337778) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.2280852838431988) q[0];
rz(-1.2903824081276227) q[0];
ry(-1.918027567895578) q[1];
rz(3.1308220368628286) q[1];
ry(0.04865068397327963) q[2];
rz(-2.479290058468136) q[2];
ry(-0.6267869739824088) q[3];
rz(-1.6544331644386876) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.6294341116292372) q[0];
rz(-0.3191958833605548) q[0];
ry(1.1257466553145719) q[1];
rz(-0.2452780546458745) q[1];
ry(2.3667491953362365) q[2];
rz(0.44276823392718223) q[2];
ry(0.19344932463730657) q[3];
rz(-2.8517293847309664) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.9332609967245659) q[0];
rz(0.8720363042524806) q[0];
ry(-0.7916166806064426) q[1];
rz(-0.45412211016012344) q[1];
ry(-0.3015912429208265) q[2];
rz(-0.0509364283604965) q[2];
ry(0.7643043674454708) q[3];
rz(2.9737902109478833) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.4793361803094136) q[0];
rz(3.027028630698226) q[0];
ry(3.1407988814857073) q[1];
rz(1.7221146879178608) q[1];
ry(0.08491506027169748) q[2];
rz(-0.8739813323560406) q[2];
ry(2.837158366516264) q[3];
rz(2.2136099180154667) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.8754076041982242) q[0];
rz(-1.6464584947153098) q[0];
ry(-0.9877941619669316) q[1];
rz(2.7781626919917324) q[1];
ry(1.5356567224688789) q[2];
rz(1.640479754954872) q[2];
ry(-2.4316627638137365) q[3];
rz(-1.5910246829949508) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.4819270597967773) q[0];
rz(-2.1392674401432905) q[0];
ry(0.8596581094396499) q[1];
rz(2.2836133206967117) q[1];
ry(-0.8081740048053082) q[2];
rz(0.46392980818357454) q[2];
ry(-1.9063409223209584) q[3];
rz(2.5968406140018696) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.560365763907485) q[0];
rz(1.2793117041285462) q[0];
ry(0.45994436086059665) q[1];
rz(1.5294535067835424) q[1];
ry(-2.5065142049018587) q[2];
rz(-2.462887757450867) q[2];
ry(-0.4686559605848499) q[3];
rz(-2.6930778700789255) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.241336354149074) q[0];
rz(-0.08944228575282502) q[0];
ry(-1.2577223082271924) q[1];
rz(-1.205689409392415) q[1];
ry(-0.3066552214009289) q[2];
rz(0.7309960091031314) q[2];
ry(2.962579185232166) q[3];
rz(-1.220403559253767) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.6022737967274647) q[0];
rz(-2.1044953359876226) q[0];
ry(-2.936237849617524) q[1];
rz(-1.5621364157968127) q[1];
ry(1.9179637984701217) q[2];
rz(-3.124177736146249) q[2];
ry(-0.23028476252257385) q[3];
rz(-1.5400373957132447) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-3.008005815793739) q[0];
rz(-1.826632588112342) q[0];
ry(0.27588690157675233) q[1];
rz(-1.4609664146200245) q[1];
ry(-2.8625739147353375) q[2];
rz(-0.4159322270447765) q[2];
ry(-1.106931980302513) q[3];
rz(-2.4544526223984353) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.7092845170267799) q[0];
rz(0.8289101244880969) q[0];
ry(-2.822651208394701) q[1];
rz(3.0490095254246743) q[1];
ry(-0.26133822027748144) q[2];
rz(2.6551076951972057) q[2];
ry(0.21478368465193143) q[3];
rz(1.5358391140592467) q[3];