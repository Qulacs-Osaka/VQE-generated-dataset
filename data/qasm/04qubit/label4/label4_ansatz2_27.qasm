OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.2439748478572215) q[0];
rz(2.4130473812608777) q[0];
ry(2.946863921305643) q[1];
rz(2.35372112879493) q[1];
ry(-1.1789745703263197) q[2];
rz(0.11388317481388022) q[2];
ry(0.1887009760845153) q[3];
rz(-2.7295102319238644) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.19290922806326147) q[0];
rz(1.9149261629407954) q[0];
ry(2.2260746528071227) q[1];
rz(2.1721010976105513) q[1];
ry(3.080086858051717) q[2];
rz(-0.8854498173344811) q[2];
ry(-0.507658804657293) q[3];
rz(-0.7889746946088857) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.0992448033299107) q[0];
rz(1.5352290020528612) q[0];
ry(2.319973642900969) q[1];
rz(-2.6210605964340914) q[1];
ry(1.2492056231868849) q[2];
rz(2.9165190875017206) q[2];
ry(0.24225290831976043) q[3];
rz(-1.081853422628709) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.8946715006935817) q[0];
rz(-0.16723980338858713) q[0];
ry(-1.1761108892388732) q[1];
rz(-0.5644692024849816) q[1];
ry(-2.492083831559537) q[2];
rz(-1.637483940103889) q[2];
ry(-3.1011958907611126) q[3];
rz(2.8678334730432438) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-3.0892302361468116) q[0];
rz(2.5228654488293207) q[0];
ry(-0.9315832003171047) q[1];
rz(-2.9749025883054543) q[1];
ry(-2.3208843379489954) q[2];
rz(1.724042341314323) q[2];
ry(-2.3670660699581325) q[3];
rz(1.3335061687802716) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.347281509789249) q[0];
rz(0.9758023691036346) q[0];
ry(-0.7925213473494694) q[1];
rz(-2.094851494061494) q[1];
ry(-0.7807491858640667) q[2];
rz(1.5032807362030807) q[2];
ry(-1.7886910003347778) q[3];
rz(-1.196920191025857) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.1520584301468357) q[0];
rz(-2.1051642147974747) q[0];
ry(0.7200398425378846) q[1];
rz(-0.39066633416892754) q[1];
ry(0.5965164795926632) q[2];
rz(-0.5180665623147518) q[2];
ry(-0.1961728244105484) q[3];
rz(-0.4327409955421411) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.762765746037255) q[0];
rz(2.8589721075374155) q[0];
ry(-2.188868931238787) q[1];
rz(-2.525974174018169) q[1];
ry(0.3005497666333206) q[2];
rz(2.398217888197792) q[2];
ry(0.5099616528597091) q[3];
rz(-3.066647383964911) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.805476174858063) q[0];
rz(1.0184667356328554) q[0];
ry(-1.1254435419621942) q[1];
rz(-0.30654620638369556) q[1];
ry(2.8730518344930513) q[2];
rz(0.015013897165498946) q[2];
ry(2.9411728720382424) q[3];
rz(-0.256389192272856) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.797310675046531) q[0];
rz(-2.8609369987523614) q[0];
ry(-0.12465995319583278) q[1];
rz(1.0893910022306497) q[1];
ry(2.375712940758798) q[2];
rz(-0.6137150057752199) q[2];
ry(1.997653272607405) q[3];
rz(-1.0021132261985937) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.058457888058472) q[0];
rz(-0.9766040188955625) q[0];
ry(-0.10223378927686166) q[1];
rz(3.037479016153601) q[1];
ry(1.7175271792871865) q[2];
rz(-1.6405157473117828) q[2];
ry(1.1624946554077953) q[3];
rz(-1.7095636508781986) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.08449729572760045) q[0];
rz(-1.324126126483372) q[0];
ry(-0.4085439461994901) q[1];
rz(-1.663309186946128) q[1];
ry(-2.6283108685402996) q[2];
rz(0.13027399245529203) q[2];
ry(2.821161553360738) q[3];
rz(-0.9447480415083671) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.9400168360054878) q[0];
rz(-2.4677418570759326) q[0];
ry(-2.1805909692641725) q[1];
rz(-1.2834501505279032) q[1];
ry(-2.2319271884529472) q[2];
rz(0.9327127969916407) q[2];
ry(2.546117610102733) q[3];
rz(0.47283387442116354) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.5923073894856516) q[0];
rz(2.9231947499168736) q[0];
ry(0.4202161658468029) q[1];
rz(-0.7821367830627777) q[1];
ry(0.8726345681264203) q[2];
rz(2.4343870920800894) q[2];
ry(0.6398611455062406) q[3];
rz(-2.6546409653340994) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.7031863972249432) q[0];
rz(-0.2396027261396654) q[0];
ry(-1.4648214836071647) q[1];
rz(0.08778200090532388) q[1];
ry(-1.9340584137990577) q[2];
rz(-2.1197829925261544) q[2];
ry(0.3686134914129795) q[3];
rz(-2.9170275525729537) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.2620717966238804) q[0];
rz(-0.3759615558148772) q[0];
ry(1.9239050817768648) q[1];
rz(-1.4563904249437876) q[1];
ry(0.5634914243989083) q[2];
rz(1.3526182024549787) q[2];
ry(0.3818066211677049) q[3];
rz(1.5683536153091497) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.7573550528985742) q[0];
rz(-1.8595136368137029) q[0];
ry(2.2252985471559845) q[1];
rz(-0.7211903160366324) q[1];
ry(-0.2009692151745055) q[2];
rz(1.143678316173587) q[2];
ry(-2.2420933502516105) q[3];
rz(0.5555161260729579) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.6959048787861224) q[0];
rz(-2.1531716385973083) q[0];
ry(-2.47580776760782) q[1];
rz(-2.0859720467737404) q[1];
ry(0.12320565204954548) q[2];
rz(3.0170844695242685) q[2];
ry(-2.9803238958483393) q[3];
rz(0.9513064435114923) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.2383617928416681) q[0];
rz(1.9204117197448667) q[0];
ry(-1.0498094412386703) q[1];
rz(-2.7035057959789195) q[1];
ry(1.0066137350362723) q[2];
rz(-2.7003348419151854) q[2];
ry(-0.7750701922365837) q[3];
rz(2.7469563749831134) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(3.1221397913791367) q[0];
rz(-0.02846088780757938) q[0];
ry(-2.912589118687864) q[1];
rz(1.024713152498418) q[1];
ry(-1.662172716815685) q[2];
rz(0.946965433763733) q[2];
ry(2.2539356618016413) q[3];
rz(-2.7340924800560336) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.6254829726727755) q[0];
rz(-0.30636816528525646) q[0];
ry(3.108103750639208) q[1];
rz(0.6770287447650728) q[1];
ry(-2.8934530805194623) q[2];
rz(-1.7338397528991498) q[2];
ry(1.3051221555326995) q[3];
rz(2.3682183531602115) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.3264988183584734) q[0];
rz(-2.8417894719687844) q[0];
ry(-2.8843603417065227) q[1];
rz(-0.7252728495915131) q[1];
ry(1.319927932228625) q[2];
rz(2.0547047096817046) q[2];
ry(0.9957477247712879) q[3];
rz(-1.350251545008292) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.11273322898935474) q[0];
rz(2.907116034835986) q[0];
ry(-1.164403715532506) q[1];
rz(-2.6249726032126977) q[1];
ry(-3.0688803595531566) q[2];
rz(-1.4976414016407338) q[2];
ry(0.2692832014431037) q[3];
rz(-2.818055154795428) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.19936791364836942) q[0];
rz(0.9441488647071284) q[0];
ry(1.2427055023455482) q[1];
rz(-1.3947689191306263) q[1];
ry(-0.5300350623718009) q[2];
rz(0.4904440929182751) q[2];
ry(1.3989497918530285) q[3];
rz(3.0709944464508694) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(3.101187347457976) q[0];
rz(-1.000783453283726) q[0];
ry(2.239245045566193) q[1];
rz(1.1039779975255741) q[1];
ry(2.956721901332695) q[2];
rz(-0.3625169017681529) q[2];
ry(2.0449697940609592) q[3];
rz(0.5212816051668387) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.2704624298002605) q[0];
rz(0.313726692717245) q[0];
ry(-2.847970733324579) q[1];
rz(-0.03141626233134415) q[1];
ry(-2.642553473569945) q[2];
rz(1.9821726115379028) q[2];
ry(-0.7630427024143106) q[3];
rz(-0.2778926088744731) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.608618339410369) q[0];
rz(2.371735281666176) q[0];
ry(3.1300681444493907) q[1];
rz(0.6895183020468802) q[1];
ry(0.9949496127329579) q[2];
rz(-2.121234102396271) q[2];
ry(-2.609346424006087) q[3];
rz(0.9667944523810643) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.6467172074065803) q[0];
rz(-2.6541071719695104) q[0];
ry(0.5293027464442497) q[1];
rz(-1.857330000468437) q[1];
ry(-0.8668867043567294) q[2];
rz(1.0233795880304097) q[2];
ry(0.33169297821405497) q[3];
rz(-1.9876057254420614) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.248148631850545) q[0];
rz(-1.3319367576702943) q[0];
ry(-2.7217782082086965) q[1];
rz(0.29064053964450304) q[1];
ry(-2.489221169676413) q[2];
rz(-2.347849012876925) q[2];
ry(1.8486066521520859) q[3];
rz(1.232675481323192) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.6408714260774548) q[0];
rz(-1.3135472520325622) q[0];
ry(1.088135644207014) q[1];
rz(-0.27521147938100765) q[1];
ry(-1.181538888001878) q[2];
rz(-2.571022394945517) q[2];
ry(-1.6393856700062104) q[3];
rz(1.3777567373248951) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.456083631482982) q[0];
rz(0.4416613095802442) q[0];
ry(0.9069788796890386) q[1];
rz(-1.9581269740250729) q[1];
ry(2.130889264354189) q[2];
rz(-2.4766790371182466) q[2];
ry(0.5411773385629308) q[3];
rz(1.5405846532811642) q[3];