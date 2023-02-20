OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.14645453873590797) q[0];
rz(-2.2122431772118425) q[0];
ry(-2.930320214427636) q[1];
rz(-2.878532969895945) q[1];
ry(-2.001008220280988) q[2];
rz(-0.7689128994972894) q[2];
ry(0.021800328576668804) q[3];
rz(0.028036395640493872) q[3];
ry(0.000383841861801848) q[4];
rz(-2.472910405809119) q[4];
ry(-1.5595330219591828) q[5];
rz(-0.40559102600322594) q[5];
ry(-3.1223325234200554) q[6];
rz(-2.3565642904580053) q[6];
ry(1.598862475162306) q[7];
rz(-0.5818286803339037) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.8918326027617303) q[0];
rz(1.206743187916305) q[0];
ry(1.9992457527838932) q[1];
rz(1.3118427837919242) q[1];
ry(2.253319210227221) q[2];
rz(0.12533103202240792) q[2];
ry(-0.0007280036999240667) q[3];
rz(-1.1820142556223332) q[3];
ry(0.4832591785297229) q[4];
rz(-0.15463450458081596) q[4];
ry(0.004824424977749264) q[5];
rz(-1.0325904430868915) q[5];
ry(1.2267333423072717) q[6];
rz(1.130648831611121) q[6];
ry(-0.44828120631467266) q[7];
rz(1.8252298650979124) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.06740394932175953) q[0];
rz(-1.1234634694513694) q[0];
ry(-1.3480495551567908) q[1];
rz(1.188788256835815) q[1];
ry(3.0081901360788854) q[2];
rz(1.5526078257077351) q[2];
ry(-0.4857377192430735) q[3];
rz(2.569515274687719) q[3];
ry(3.119991723509535) q[4];
rz(-2.7315157974507938) q[4];
ry(-0.07494488689857458) q[5];
rz(-2.964646825473186) q[5];
ry(0.02856120750996283) q[6];
rz(-2.0230944488506575) q[6];
ry(0.6440357015872451) q[7];
rz(0.10834795773281536) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.475677959452265) q[0];
rz(2.4066262736699024) q[0];
ry(-0.026552125813842524) q[1];
rz(0.2974997397813235) q[1];
ry(0.6944371198206606) q[2];
rz(2.299865575902815) q[2];
ry(-0.00017970893325358617) q[3];
rz(-2.9744997027333553) q[3];
ry(2.760306494193036) q[4];
rz(-1.5035175251641322) q[4];
ry(-0.0027396183930221696) q[5];
rz(2.302528550882979) q[5];
ry(-0.021051741509106655) q[6];
rz(-0.9638940785998643) q[6];
ry(-2.2958316113451382) q[7];
rz(-1.818721742805474) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-3.0503508339893526) q[0];
rz(3.117774593140709) q[0];
ry(-1.2853769430029673) q[1];
rz(-0.8548605221298313) q[1];
ry(2.9643953450667064) q[2];
rz(0.7437408311622234) q[2];
ry(-2.7876094180848603) q[3];
rz(-1.9161119357160301) q[3];
ry(3.0078060534842996) q[4];
rz(-0.27633257803559275) q[4];
ry(-1.6909120367970283) q[5];
rz(-0.43834502573284634) q[5];
ry(-2.9527229689158108) q[6];
rz(2.8485432865014455) q[6];
ry(1.7364609249019534) q[7];
rz(-0.9835280105712286) q[7];