OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.788438649948965) q[0];
rz(-0.0983760637708298) q[0];
ry(1.0396257883682036) q[1];
rz(-2.1999895592235967) q[1];
ry(1.9104910215795385) q[2];
rz(-1.6136003095387297) q[2];
ry(1.2549450078391056) q[3];
rz(2.2379962382393472) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.8611325838815436) q[0];
rz(0.8322626004809033) q[0];
ry(-2.3993608421365957) q[1];
rz(-1.8594712961622442) q[1];
ry(2.798233972448097) q[2];
rz(2.0707568337988915) q[2];
ry(0.30240338678236833) q[3];
rz(-3.0324707924361802) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.30908289044196) q[0];
rz(-2.222997682454919) q[0];
ry(2.685541563510492) q[1];
rz(1.6768810981215498) q[1];
ry(0.3504101835780227) q[2];
rz(1.8291271403092924) q[2];
ry(2.5120794273919382) q[3];
rz(-0.7425492096234665) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.9746576385809789) q[0];
rz(-1.2558011879130726) q[0];
ry(-0.938469051207159) q[1];
rz(1.3676251110888202) q[1];
ry(-2.9047882728261514) q[2];
rz(-0.9807666514357667) q[2];
ry(0.05447012281133379) q[3];
rz(-0.16945358277510378) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.330003700891998) q[0];
rz(-2.741137741359638) q[0];
ry(0.29601415630776984) q[1];
rz(-0.5177142820348252) q[1];
ry(1.2055917691519724) q[2];
rz(-0.639736144029854) q[2];
ry(-2.492001769042842) q[3];
rz(1.6103850379784408) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.0110934250306949) q[0];
rz(0.8504615730483165) q[0];
ry(-1.9855405306695078) q[1];
rz(1.7642162994562005) q[1];
ry(-2.0006217925356085) q[2];
rz(-2.4404913092050537) q[2];
ry(-2.65163973647556) q[3];
rz(2.4302912695581798) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.057717126229230864) q[0];
rz(3.102878701741009) q[0];
ry(1.2544954922090739) q[1];
rz(-1.1145745899725685) q[1];
ry(-2.9857426958010227) q[2];
rz(1.3043263684138466) q[2];
ry(-0.42381620952616417) q[3];
rz(0.8559192540574143) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.3693321685196747) q[0];
rz(1.8737296274950284) q[0];
ry(2.0293721720273705) q[1];
rz(-1.4117422064539955) q[1];
ry(0.8426520784862207) q[2];
rz(-1.8554497061301092) q[2];
ry(0.5859780860928755) q[3];
rz(-0.164930108188301) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.693858876716675) q[0];
rz(2.943655747143759) q[0];
ry(-1.469949917466403) q[1];
rz(1.6417321202237198) q[1];
ry(0.14679222944426418) q[2];
rz(-0.5436338951653107) q[2];
ry(0.28156424182086237) q[3];
rz(-1.936730176745729) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.4412346042703303) q[0];
rz(1.9120212765859632) q[0];
ry(0.17255191815806978) q[1];
rz(-0.045208139754786984) q[1];
ry(1.9172446783995545) q[2];
rz(0.9629276425064823) q[2];
ry(-3.049365812910895) q[3];
rz(-2.712684135990205) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.5244276067833902) q[0];
rz(1.401563192381572) q[0];
ry(-2.21398738866324) q[1];
rz(-1.0359587953680975) q[1];
ry(2.401721888299237) q[2];
rz(-2.958025100464562) q[2];
ry(-1.464497588095461) q[3];
rz(-0.21741142998878726) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.1356849132782099) q[0];
rz(2.066740949609005) q[0];
ry(0.9117633842187381) q[1];
rz(-2.2740702976816163) q[1];
ry(-0.2381644253965314) q[2];
rz(-2.5796609817534284) q[2];
ry(-0.5550630681206614) q[3];
rz(-1.2012802652317704) q[3];