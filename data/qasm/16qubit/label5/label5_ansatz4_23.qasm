OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.4330316032962784) q[0];
rz(2.1843672135457504) q[0];
ry(-1.3857315063195443) q[1];
rz(-2.4627523827734565) q[1];
ry(-0.005763676853128345) q[2];
rz(-1.2644504428717758) q[2];
ry(-0.004256426781923659) q[3];
rz(2.08384982671259) q[3];
ry(1.5702541253300142) q[4];
rz(-0.03613683115596977) q[4];
ry(1.5695034218741033) q[5];
rz(1.0944203997014483) q[5];
ry(1.4450814751550847) q[6];
rz(-0.24637752205937985) q[6];
ry(-2.329150441021815) q[7];
rz(-1.8078999873561417) q[7];
ry(-0.7044216762347917) q[8];
rz(-3.1130046494321717) q[8];
ry(0.6990290167436292) q[9];
rz(-1.6244394045312704) q[9];
ry(-0.0001435176865404995) q[10];
rz(-0.6481852089895261) q[10];
ry(2.3205482358767697) q[11];
rz(1.967583887506059) q[11];
ry(3.1404659657798506) q[12];
rz(-1.742367131891701) q[12];
ry(3.1411955834282015) q[13];
rz(-0.8772877706423654) q[13];
ry(1.5606309778443117) q[14];
rz(-2.5661394104522834) q[14];
ry(-1.5810186741505994) q[15];
rz(1.5227730179386922) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(3.037799304644984) q[0];
rz(-0.510852614814314) q[0];
ry(-3.011485712090033) q[1];
rz(2.226320384559083) q[1];
ry(1.9904064849731968) q[2];
rz(-1.1630978252264697) q[2];
ry(2.0757773812529985) q[3];
rz(2.2985168824285704) q[3];
ry(3.1131734368229194) q[4];
rz(-1.7576249894623859) q[4];
ry(-3.141518181869589) q[5];
rz(2.022561027790755) q[5];
ry(3.0412654943075497) q[6];
rz(2.9460337702381083) q[6];
ry(-2.823947892994979) q[7];
rz(2.771831235505357) q[7];
ry(-0.9404849319595634) q[8];
rz(2.644828586314499) q[8];
ry(1.2661394046257257) q[9];
rz(-1.6871509593944838) q[9];
ry(3.10823436766583e-05) q[10];
rz(-2.8422293253664708) q[10];
ry(-1.0575269498167534) q[11];
rz(0.3516207553573336) q[11];
ry(1.5767009046786884) q[12];
rz(1.2503917652589678) q[12];
ry(1.5781809068290356) q[13];
rz(-1.6115597461933717) q[13];
ry(1.4556786883089465) q[14];
rz(1.827517820406854) q[14];
ry(-2.5071381694032335) q[15];
rz(-0.5405226462784469) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.9689224667861196) q[0];
rz(-1.2652287862224352) q[0];
ry(-1.2872371419424926) q[1];
rz(0.9593277149271675) q[1];
ry(3.1383428579987607) q[2];
rz(-2.0583679034838447) q[2];
ry(-0.005718626695312423) q[3];
rz(-0.5657765006934142) q[3];
ry(0.0022164493007323927) q[4];
rz(1.7276979764009195) q[4];
ry(3.140151939175657) q[5];
rz(-2.212591756821303) q[5];
ry(1.0108059683547532) q[6];
rz(0.4720595628631834) q[6];
ry(-1.2237736148934413) q[7];
rz(1.2649507368945665) q[7];
ry(-2.689830665635486) q[8];
rz(1.4268988448291695) q[8];
ry(0.6615389804742253) q[9];
rz(-2.5854442553063977) q[9];
ry(0.0004610608886652656) q[10];
rz(1.140142344154503) q[10];
ry(3.141224797989967) q[11];
rz(-0.715017278024388) q[11];
ry(-1.7197238452270946) q[12];
rz(2.8022274309746944) q[12];
ry(1.8590923213823762) q[13];
rz(-1.1972683511167146) q[13];
ry(-0.010568665666269957) q[14];
rz(-1.8706749852588584) q[14];
ry(-2.484026997960691) q[15];
rz(1.181943385482298) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.28493847212329904) q[0];
rz(-0.032365728418838606) q[0];
ry(-2.653941452130089) q[1];
rz(-2.3749648480372842) q[1];
ry(2.5564474332611886) q[2];
rz(-1.0540346194286236) q[2];
ry(2.7678013419729424) q[3];
rz(-1.4923723931375505) q[3];
ry(0.21725694914828073) q[4];
rz(-3.0068412225813526) q[4];
ry(-1.7733686760048644) q[5];
rz(2.058324176485439) q[5];
ry(1.2305884741471464) q[6];
rz(-2.206097731284805) q[6];
ry(1.8908594077514227) q[7];
rz(2.9535090592865063) q[7];
ry(1.4041065058207507) q[8];
rz(-3.0744004257629003) q[8];
ry(-1.9131116862258777) q[9];
rz(-1.749608490127173) q[9];
ry(-3.135487838906273) q[10];
rz(0.5532751349809082) q[10];
ry(0.00036279401153116453) q[11];
rz(-2.5618397385260936) q[11];
ry(-3.141211192630328) q[12];
rz(2.961043350391988) q[12];
ry(-0.009190483980759723) q[13];
rz(1.366837534507974) q[13];
ry(0.015850470659036553) q[14];
rz(1.9651600926414765) q[14];
ry(-0.01986418895989606) q[15];
rz(1.8836859530426802) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.673299173422841) q[0];
rz(-1.0716801893133043) q[0];
ry(-3.1046457323007854) q[1];
rz(-1.6768344693067643) q[1];
ry(-2.1116276263447644) q[2];
rz(1.6069295143600626) q[2];
ry(-2.3903049829004326) q[3];
rz(0.9641946959297609) q[3];
ry(3.1414484037144392) q[4];
rz(-3.046177651557802) q[4];
ry(-3.133757966423018) q[5];
rz(2.1556860288547224) q[5];
ry(0.46859306964318703) q[6];
rz(0.4382592780127943) q[6];
ry(-2.600020440240722) q[7];
rz(1.5070744563942728) q[7];
ry(-0.34618425802261493) q[8];
rz(2.2894980832404648) q[8];
ry(3.0865779612887327) q[9];
rz(-1.1635113236308101) q[9];
ry(1.5694012101565322) q[10];
rz(1.5183413152827865) q[10];
ry(1.5708182397824781) q[11];
rz(-0.01639755629357342) q[11];
ry(1.3738722462654378) q[12];
rz(0.29281786298670676) q[12];
ry(1.2928192069648485) q[13];
rz(-1.0077025470017913) q[13];
ry(-3.086054638902904) q[14];
rz(-3.090110046792621) q[14];
ry(2.43828285942694) q[15];
rz(0.7695556385986501) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.6126662046622) q[0];
rz(2.2982649733045344) q[0];
ry(2.869286240643055) q[1];
rz(-3.100179275082128) q[1];
ry(1.0625406604829992) q[2];
rz(-1.7183105867044517) q[2];
ry(3.0530272713115805) q[3];
rz(-0.9023082691791914) q[3];
ry(0.018804297038023776) q[4];
rz(0.04401771440059991) q[4];
ry(3.125541898639868) q[5];
rz(-3.0444356815230673) q[5];
ry(-1.543452948332155) q[6];
rz(-0.8396934449167935) q[6];
ry(0.9743321849145178) q[7];
rz(-2.119757488107509) q[7];
ry(-3.140718257397125) q[8];
rz(3.1373979788086457) q[8];
ry(-0.00037337842210227734) q[9];
rz(2.4752318109236544) q[9];
ry(0.9983561498667052) q[10];
rz(-2.7583773249137153) q[10];
ry(0.03127290981952768) q[11];
rz(1.5876881565008905) q[11];
ry(-1.5703932915116765) q[12];
rz(1.590193311129906) q[12];
ry(0.0012823284947742764) q[13];
rz(-0.7649154067063987) q[13];
ry(-1.415256516357041) q[14];
rz(-0.6206931374202084) q[14];
ry(-1.7616401984019028) q[15];
rz(-3.1252270667643205) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.570535031198287) q[0];
rz(1.7805131692868779) q[0];
ry(-1.3242671449284265) q[1];
rz(1.0733543565889656) q[1];
ry(-0.5800126801927874) q[2];
rz(-2.1774828876184475) q[2];
ry(2.062841745212331) q[3];
rz(0.8520869265451809) q[3];
ry(1.5728150742682336) q[4];
rz(-2.546810666027685) q[4];
ry(1.5697746284746303) q[5];
rz(0.5993453367395638) q[5];
ry(-1.342574861212869) q[6];
rz(-2.7495251124396964) q[6];
ry(-1.3699890550110903) q[7];
rz(-1.557891249354877) q[7];
ry(-2.1403396849855234) q[8];
rz(-1.2802252606199742) q[8];
ry(1.9251120462611064) q[9];
rz(-2.837660233428603) q[9];
ry(3.140364879718655) q[10];
rz(0.2390158467067813) q[10];
ry(1.5714242484009047) q[11];
rz(0.3374869708508612) q[11];
ry(1.8843280013591248) q[12];
rz(1.8601201183445624) q[12];
ry(-0.08815939679134961) q[13];
rz(1.6417874089003757) q[13];
ry(1.845473275921853) q[14];
rz(-3.1354907982594815) q[14];
ry(-1.2940467348699036) q[15];
rz(2.9249562739477457) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.3769277089315652) q[0];
rz(1.4655391155972517) q[0];
ry(2.7916631381451738) q[1];
rz(-1.53338820672408) q[1];
ry(1.1714207274205124) q[2];
rz(-1.411174368007913) q[2];
ry(1.3879052837461407) q[3];
rz(-1.4347503189724797) q[3];
ry(1.5717488064846594) q[4];
rz(-0.13202956920716924) q[4];
ry(1.5696309570486937) q[5];
rz(1.99294099827641) q[5];
ry(0.20386267410541814) q[6];
rz(2.942564828110028) q[6];
ry(2.587413338046688) q[7];
rz(-2.1834251829954696) q[7];
ry(-1.4469429962146867) q[8];
rz(-1.00220049689601) q[8];
ry(1.8083556635240097) q[9];
rz(0.8245822467880172) q[9];
ry(3.1415337382277295) q[10];
rz(1.92610314223929) q[10];
ry(0.00042456435309706694) q[11];
rz(-1.4198423468838928) q[11];
ry(-0.015052842108108422) q[12];
rz(1.635270529385436) q[12];
ry(2.983879195868419) q[13];
rz(-1.5595350972608157) q[13];
ry(0.6450056923325276) q[14];
rz(2.772035717166064) q[14];
ry(1.3604481654002454) q[15];
rz(3.065706135593389) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.8310137389836814) q[0];
rz(0.4014018314968011) q[0];
ry(-2.4623418017952123) q[1];
rz(-1.0850977652667728) q[1];
ry(-2.9968404764558567) q[2];
rz(0.7500872206429915) q[2];
ry(-3.0706551056263787) q[3];
rz(-2.787764342091525) q[3];
ry(-0.0012639227255380624) q[4];
rz(3.0501301244862264) q[4];
ry(-3.141159707146711) q[5];
rz(0.42102626784690916) q[5];
ry(-3.0489478250514312) q[6];
rz(0.9447799173621843) q[6];
ry(1.218437484685711) q[7];
rz(-0.14806961147285014) q[7];
ry(-1.5172299660716435) q[8];
rz(-0.9560136141306828) q[8];
ry(1.5583571699374186) q[9];
rz(-2.8952149739107553) q[9];
ry(1.5700783907751665) q[10];
rz(-1.946264377702068) q[10];
ry(1.5724319186660465) q[11];
rz(-3.140409811760283) q[11];
ry(-3.134467175362781) q[12];
rz(-1.6568618062201252) q[12];
ry(-3.026975968777877) q[13];
rz(-1.5938579896543805) q[13];
ry(-0.4030446372527266) q[14];
rz(-1.5946507427486734) q[14];
ry(-2.0018987614524395) q[15];
rz(-0.7068482640664685) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.6496954108815112) q[0];
rz(-1.1097826787255647) q[0];
ry(1.4536525084840353) q[1];
rz(0.784401899438002) q[1];
ry(0.986359411073388) q[2];
rz(2.328337869344169) q[2];
ry(2.053203656909496) q[3];
rz(0.3879219676257633) q[3];
ry(3.141074625900341) q[4];
rz(-0.2231861892966966) q[4];
ry(0.4708219166148613) q[5];
rz(-1.5696654552726008) q[5];
ry(-1.6786283022473847) q[6];
rz(-1.409727737162988) q[6];
ry(-2.417514663476836) q[7];
rz(1.1293634674908875) q[7];
ry(1.8761972145160728) q[8];
rz(-1.638503223453832) q[8];
ry(-0.3201268271381941) q[9];
rz(2.9034889416866045) q[9];
ry(-0.00020938058763864118) q[10];
rz(1.9799180968405334) q[10];
ry(-1.6040854751322513) q[11];
rz(1.7029365801831842) q[11];
ry(3.1283692680032913) q[12];
rz(-0.0046967122188883445) q[12];
ry(-0.00970378384118664) q[13];
rz(0.12035066976639998) q[13];
ry(-2.1206024632758926) q[14];
rz(-0.5445919302880222) q[14];
ry(-1.7129981028410206) q[15];
rz(-1.7691708896846743) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.997969264060666) q[0];
rz(0.24798977448344853) q[0];
ry(-2.839990707231024) q[1];
rz(3.081435034374367) q[1];
ry(-0.021772827341211533) q[2];
rz(2.442864233138482) q[2];
ry(-0.1923694814801289) q[3];
rz(2.2104913729240905) q[3];
ry(1.2559033029519988) q[4];
rz(-1.4836037000545987) q[4];
ry(-1.2560956119032656) q[5];
rz(1.585681857517173) q[5];
ry(-1.5796753183444345) q[6];
rz(1.5341741788439052) q[6];
ry(-1.6581022793125122) q[7];
rz(1.6288339459023655) q[7];
ry(0.0287721575083566) q[8];
rz(-1.0807590564561054) q[8];
ry(3.112585980495274) q[9];
rz(0.6689771863734855) q[9];
ry(-0.00025318050837075384) q[10];
rz(-0.7796218806178932) q[10];
ry(3.112189184243951) q[11];
rz(-1.3746869957779921) q[11];
ry(0.03375693503457544) q[12];
rz(1.7539120496560225) q[12];
ry(-1.4224189581997824) q[13];
rz(2.1847652968246383) q[13];
ry(2.715680575894449) q[14];
rz(2.5901832041034325) q[14];
ry(2.666565593493116) q[15];
rz(1.9951614201635826) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.280298024618256) q[0];
rz(-1.3649519356188264) q[0];
ry(-2.0610155077346626) q[1];
rz(2.427802082614562) q[1];
ry(-0.45328556926727964) q[2];
rz(-2.6366171126537292) q[2];
ry(-0.30223144358464094) q[3];
rz(1.8206831391249714) q[3];
ry(0.025812630677508608) q[4];
rz(-0.08662084111314526) q[4];
ry(-3.1202495331056346) q[5];
rz(0.014682362526879269) q[5];
ry(-1.2125691261079412) q[6];
rz(-0.11109312905033963) q[6];
ry(-1.4759900822054401) q[7];
rz(0.09918796724966787) q[7];
ry(2.724651365482691) q[8];
rz(-1.2892565020022686) q[8];
ry(1.5142669315325596) q[9];
rz(2.0510607544995976) q[9];
ry(-0.000349676019827605) q[10];
rz(2.3120561292235315) q[10];
ry(-0.023102489159187993) q[11];
rz(1.4855528312053767) q[11];
ry(3.138376663035639) q[12];
rz(-0.98186509559984) q[12];
ry(-0.004433218466898632) q[13];
rz(-3.0335630729295335) q[13];
ry(2.155607112637883) q[14];
rz(1.1830726690604914) q[14];
ry(-0.02833109583419091) q[15];
rz(1.4226210215252275) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.1062736345938164) q[0];
rz(0.6208589976236043) q[0];
ry(1.406364152439782) q[1];
rz(2.3739285722684054) q[1];
ry(0.2623439737950038) q[2];
rz(-1.4944070125728732) q[2];
ry(-0.17439906788499945) q[3];
rz(-0.07647728867703185) q[3];
ry(1.5705650862606382) q[4];
rz(-0.6843904033680186) q[4];
ry(-1.5714998164698226) q[5];
rz(-2.245879093328621) q[5];
ry(-0.21429731736632007) q[6];
rz(-2.9877006459441753) q[6];
ry(0.7445056511758456) q[7];
rz(-0.1392443668877448) q[7];
ry(0.6044435628177478) q[8];
rz(0.014589079300099285) q[8];
ry(0.6073125651856115) q[9];
rz(-1.5171659665728159) q[9];
ry(3.1302383121441717) q[10];
rz(3.137525492407632) q[10];
ry(-3.133639174576402) q[11];
rz(-0.021145918490991455) q[11];
ry(-1.6027752338406893) q[12];
rz(-2.0441352039917335) q[12];
ry(1.3443274179240525) q[13];
rz(0.02692654344989094) q[13];
ry(2.9607756219884314) q[14];
rz(0.9813795117631222) q[14];
ry(-1.4189554684181287) q[15];
rz(1.8487314872668508) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.006475046868561724) q[0];
rz(-1.8894899889814145) q[0];
ry(-1.5202249689352956) q[1];
rz(-0.8828675103623148) q[1];
ry(3.075044400761575) q[2];
rz(-2.9289347287484344) q[2];
ry(-0.2674304010818196) q[3];
rz(0.08848869544298603) q[3];
ry(-3.1414116882266367) q[4];
rz(1.1864367972120515) q[4];
ry(0.0013443197985507638) q[5];
rz(-0.5547241788806775) q[5];
ry(1.5622717831903203) q[6];
rz(1.839081960488278) q[6];
ry(-1.6079410326212955) q[7];
rz(0.08676083734896611) q[7];
ry(-1.546642923610938) q[8];
rz(-1.83864125847287) q[8];
ry(-2.6293381989297377) q[9];
rz(-1.593661648656055) q[9];
ry(-1.5709223029388875) q[10];
rz(-0.00792406516211841) q[10];
ry(-1.5728772372990383) q[11];
rz(3.1319119949403516) q[11];
ry(-3.125320917469694) q[12];
rz(-1.7122157492939838) q[12];
ry(-0.03251082212887191) q[13];
rz(-1.344608238908772) q[13];
ry(3.1400389853515476) q[14];
rz(1.8811992032785687) q[14];
ry(3.138266314052457) q[15];
rz(0.4152046832127924) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.6923916091125766) q[0];
rz(1.8223596639140336) q[0];
ry(2.1006248031440715) q[1];
rz(-2.201772404232581) q[1];
ry(-1.6207494649928735) q[2];
rz(3.04830119965922) q[2];
ry(1.9411092297992463) q[3];
rz(0.5531352773885487) q[3];
ry(-3.1414321774353255) q[4];
rz(-2.6153327039815513) q[4];
ry(-0.0006558973470482954) q[5];
rz(-1.4167345379982228) q[5];
ry(1.580909044115102) q[6];
rz(-0.2354630368534858) q[6];
ry(1.5364433507572217) q[7];
rz(0.6589011616530511) q[7];
ry(-2.64398197219963) q[8];
rz(-2.9817459635528163) q[8];
ry(-3.132036276778612) q[9];
rz(1.4961589380630063) q[9];
ry(1.570587677375315) q[10];
rz(1.1829669252732788) q[10];
ry(1.570489151828083) q[11];
rz(1.9011136699866495) q[11];
ry(-3.083676309570549) q[12];
rz(1.4795469666416192) q[12];
ry(-1.560600656855497) q[13];
rz(1.5625231510114217) q[13];
ry(1.5006435309699535) q[14];
rz(-1.4015437188618591) q[14];
ry(-0.1450188635954257) q[15];
rz(0.2375400610006401) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.0078483988773197) q[0];
rz(1.497116968683621) q[0];
ry(3.0272788498445116) q[1];
rz(-0.6604284940047266) q[1];
ry(-1.2632049582244982) q[2];
rz(2.5780072203403) q[2];
ry(-0.8249161082067957) q[3];
rz(-2.1301311934321316) q[3];
ry(-3.138878538865089) q[4];
rz(1.3209186481374768) q[4];
ry(3.140685809832271) q[5];
rz(-0.7792147964013953) q[5];
ry(1.573326535558574) q[6];
rz(-3.10707836978141) q[6];
ry(-0.012562517392002981) q[7];
rz(2.4889958376158523) q[7];
ry(3.1346026108844787) q[8];
rz(-1.9294821123254495) q[8];
ry(-2.524828886632279) q[9];
rz(-1.8633183802805915) q[9];
ry(3.140958713096112) q[10];
rz(-0.3876225463648457) q[10];
ry(-3.141185568583746) q[11];
rz(0.40183341934062394) q[11];
ry(-1.5738067154076223) q[12];
rz(-1.3094300542925854) q[12];
ry(-1.5715296228239655) q[13];
rz(1.5690981955125602) q[13];
ry(-1.3297573246602772) q[14];
rz(-0.459431254439969) q[14];
ry(-3.137474858288804) q[15];
rz(1.4228371726617) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.4671020080863286) q[0];
rz(1.053353587948865) q[0];
ry(-2.6059589125273006) q[1];
rz(-2.2884710825227814) q[1];
ry(-2.925340025392443) q[2];
rz(-0.6887134170458865) q[2];
ry(2.9418032331959316) q[3];
rz(1.169569325485825) q[3];
ry(-9.32243787695503e-05) q[4];
rz(-0.06804556654294873) q[4];
ry(3.1384598610750722) q[5];
rz(1.4918268482494677) q[5];
ry(1.6216307047321754) q[6];
rz(-0.8172966607610587) q[6];
ry(1.5605059993254726) q[7];
rz(1.962454533776353) q[7];
ry(-2.8493878497807894) q[8];
rz(0.24954252021629117) q[8];
ry(3.104370835672632) q[9];
rz(-0.1054747251822663) q[9];
ry(1.3200796680269686) q[10];
rz(-1.3396757004317288) q[10];
ry(-3.1395578148585734) q[11];
rz(-2.7970004405713698) q[11];
ry(3.1294925812236127) q[12];
rz(-1.2806842711105864) q[12];
ry(1.643893678845805) q[13];
rz(1.5167400813505365) q[13];
ry(3.141184817580495) q[14];
rz(2.5577422558086105) q[14];
ry(1.5655379035637227) q[15];
rz(1.7368613853930195) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.066489810830734) q[0];
rz(-1.029617546252118) q[0];
ry(2.7428802380889183) q[1];
rz(-1.343117014759244) q[1];
ry(1.5416057078021321) q[2];
rz(-2.032918827271808) q[2];
ry(-2.3265312869259605) q[3];
rz(1.1738591157013296) q[3];
ry(-0.050897865206796004) q[4];
rz(2.878513707213106) q[4];
ry(-3.140953648981486) q[5];
rz(2.122258029577704) q[5];
ry(0.05422425433294987) q[6];
rz(0.8582682639869967) q[6];
ry(-0.20211794712397663) q[7];
rz(-0.231846608853777) q[7];
ry(-2.163212614915909) q[8];
rz(3.1354559262371864) q[8];
ry(1.6053263775991544) q[9];
rz(2.33746467382003) q[9];
ry(-3.1334464941848537) q[10];
rz(2.37259479432415) q[10];
ry(-3.141047181339416) q[11];
rz(0.24932067681715342) q[11];
ry(1.3707315619795872) q[12];
rz(-1.4822977203728467) q[12];
ry(0.0050424839607243754) q[13];
rz(2.2012793422393866) q[13];
ry(3.1292303333466513) q[14];
rz(-1.399743960041161) q[14];
ry(3.1280221853722567) q[15];
rz(2.975602110174139) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.0538086958667705) q[0];
rz(2.0453715154177052) q[0];
ry(0.466633836206832) q[1];
rz(0.4787559684831759) q[1];
ry(1.578653531257302) q[2];
rz(1.5126348360452644) q[2];
ry(-1.5197217921235773) q[3];
rz(3.0225341990176133) q[3];
ry(-3.1394497309236) q[4];
rz(0.4459730431486032) q[4];
ry(-3.141462641863049) q[5];
rz(-1.5540349693578652) q[5];
ry(-3.1358424594876353) q[6];
rz(2.498134962251002) q[6];
ry(3.0923203932241323) q[7];
rz(1.4205522299740494) q[7];
ry(-2.256964881749138) q[8];
rz(3.1264188455593667) q[8];
ry(0.1310958137608932) q[9];
rz(1.477483905016805) q[9];
ry(0.0030723392173203656) q[10];
rz(0.8531426833825871) q[10];
ry(-3.113110470766771) q[11];
rz(-1.4931588676771748) q[11];
ry(-2.7656441193042562) q[12];
rz(-2.7124041618666093) q[12];
ry(-3.106106085740103) q[13];
rz(0.9493085993712777) q[13];
ry(0.16588458302104933) q[14];
rz(3.0080293417385815) q[14];
ry(-3.1204576163283657) q[15];
rz(-0.3406865853814196) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.641625143846651) q[0];
rz(2.2745035596603764) q[0];
ry(1.596955383611676) q[1];
rz(-0.21904605838794833) q[1];
ry(-0.810879857600514) q[2];
rz(3.122307335617061) q[2];
ry(1.784633763569933) q[3];
rz(1.8982586998147033) q[3];
ry(-0.0006743559154361099) q[4];
rz(2.0346753602710996) q[4];
ry(-3.1367057440398662) q[5];
rz(1.3497651548494405) q[5];
ry(-0.012979331249998895) q[6];
rz(-1.0990017323354246) q[6];
ry(0.16124647699354858) q[7];
rz(1.6181538252484267) q[7];
ry(0.947002428508541) q[8];
rz(3.1347688818331196) q[8];
ry(3.1071941246164037) q[9];
rz(-0.8904159395981922) q[9];
ry(0.003377982581969974) q[10];
rz(1.718911792750605) q[10];
ry(-0.0026704516090818586) q[11];
rz(-1.6721186649662485) q[11];
ry(0.03038477058619815) q[12];
rz(-0.40603886060604744) q[12];
ry(3.138843008328315) q[13];
rz(-1.1939675199628488) q[13];
ry(-0.0004997604232634896) q[14];
rz(0.19160452616294085) q[14];
ry(3.1297345559115777) q[15];
rz(2.0010465545418343) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.0510207320831064) q[0];
rz(1.2732546790429526) q[0];
ry(-3.135206010761602) q[1];
rz(3.029562348857074) q[1];
ry(-1.5878488134457744) q[2];
rz(3.029497846794807) q[2];
ry(1.5521279474557108) q[3];
rz(3.0784202086575903) q[3];
ry(1.568888349464701) q[4];
rz(-3.1082474493289376) q[4];
ry(-1.561919665043157) q[5];
rz(3.1065876894704263) q[5];
ry(-1.4573612379757883) q[6];
rz(-3.140768340587457) q[6];
ry(1.5439727789148874) q[7];
rz(1.3265461484908299) q[7];
ry(-2.273333848963594) q[8];
rz(2.7622734216239433) q[8];
ry(1.618995628333671) q[9];
rz(-0.46110805769088525) q[9];
ry(0.481855207572794) q[10];
rz(2.316649679561464) q[10];
ry(-2.14449008380507) q[11];
rz(-1.8256756106565852) q[11];
ry(0.3953998014270271) q[12];
rz(-3.07017557588929) q[12];
ry(1.5499692239837886) q[13];
rz(-1.603178430522819) q[13];
ry(-3.007432720498623) q[14];
rz(2.024481220448604) q[14];
ry(3.0951654294540303) q[15];
rz(-1.0796946091831516) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.854393949797284) q[0];
rz(1.3337284401460348) q[0];
ry(1.1909113652629095) q[1];
rz(0.2732192312225763) q[1];
ry(3.1284545357527542) q[2];
rz(-0.38611132241349994) q[2];
ry(-1.59930229612333) q[3];
rz(1.1532390108236932) q[3];
ry(-3.1240437212223306) q[4];
rz(0.029863422128702407) q[4];
ry(3.121131433036939) q[5];
rz(3.1064532508187774) q[5];
ry(-1.5706540819335997) q[6];
rz(-0.0007498725934202955) q[6];
ry(3.1409191784958064) q[7];
rz(1.3201737153555957) q[7];
ry(-0.0022647809659117613) q[8];
rz(-1.2720245922146536) q[8];
ry(0.02175351091057076) q[9];
rz(2.0683747862257844) q[9];
ry(3.1414197256917324) q[10];
rz(0.6120031833942443) q[10];
ry(-3.1412195429871757) q[11];
rz(0.37929641386374374) q[11];
ry(-1.54408356561457) q[12];
rz(0.19541923186597865) q[12];
ry(-1.5909812663875402) q[13];
rz(3.1330514379335233) q[13];
ry(-1.1634069525700328) q[14];
rz(-0.8446439429121453) q[14];
ry(-0.006847200192362735) q[15];
rz(-1.6169475371208522) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.5895370434243725) q[0];
rz(0.4559178862879802) q[0];
ry(0.02216516945190694) q[1];
rz(1.4373462406003918) q[1];
ry(3.1011409234111573) q[2];
rz(1.4743231702524033) q[2];
ry(2.9449657787288155) q[3];
rz(2.718329505993445) q[3];
ry(-1.5475380291427354) q[4];
rz(-1.5453561849892061) q[4];
ry(1.5724541332486033) q[5];
rz(1.543336050679306) q[5];
ry(1.4570263904990481) q[6];
rz(-2.7739376188792546) q[6];
ry(1.5739360595797232) q[7];
rz(-1.5818154674660472) q[7];
ry(1.5666293888032186) q[8];
rz(0.5882066985952602) q[8];
ry(-3.1254814053862097) q[9];
rz(-1.7585514178080386) q[9];
ry(-3.1336930658531754) q[10];
rz(3.1213368077262933) q[10];
ry(-3.131200435728827) q[11];
rz(1.9463513594263737) q[11];
ry(3.118551967766885) q[12];
rz(-1.7083562403824066) q[12];
ry(-1.5732542175123054) q[13];
rz(0.6414117299111171) q[13];
ry(-0.0013571629559514518) q[14];
rz(-0.2256208486075062) q[14];
ry(-1.5650264102024254) q[15];
rz(-0.0007416378357242431) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.016564914857194246) q[0];
rz(-0.530710457904775) q[0];
ry(-0.02932035228322799) q[1];
rz(1.9936837945894432) q[1];
ry(-1.5717676617994825) q[2];
rz(1.6079768497173257) q[2];
ry(-1.6797919257561924) q[3];
rz(2.8726690066678806) q[3];
ry(-1.5669365459778906) q[4];
rz(1.5702228961141431) q[4];
ry(-1.5710400510396505) q[5];
rz(1.6040776833346309) q[5];
ry(-0.19077600077944634) q[6];
rz(-0.3550160640107487) q[6];
ry(0.08598994415656502) q[7];
rz(1.615740248101934) q[7];
ry(0.09856415823250943) q[8];
rz(-0.5897398132082019) q[8];
ry(-0.06523862793470142) q[9];
rz(-1.7212537135802044) q[9];
ry(0.0020014518316642206) q[10];
rz(-1.6856270597162375) q[10];
ry(-3.1415836232127736) q[11];
rz(1.3118932223646809) q[11];
ry(-0.002401672447345451) q[12];
rz(-2.803270594338647) q[12];
ry(3.135769105449378) q[13];
rz(2.186078239082475) q[13];
ry(-3.055880838471712) q[14];
rz(2.7423946773635226) q[14];
ry(1.5689373499559416) q[15];
rz(-3.088449309761744) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.613517442993116) q[0];
rz(3.0896821026107113) q[0];
ry(3.138555585977942) q[1];
rz(0.7378794974604558) q[1];
ry(-1.573227400419741) q[2];
rz(-1.4227550300979548) q[2];
ry(-0.014178563692142987) q[3];
rz(0.20672623821051475) q[3];
ry(1.5704781558311816) q[4];
rz(0.5028082508174981) q[4];
ry(3.1352070132261427) q[5];
rz(0.6093455960384094) q[5];
ry(1.5741044539696942) q[6];
rz(-1.5690914339257818) q[6];
ry(1.5876592612293823) q[7];
rz(-1.57325921951091) q[7];
ry(1.5677054436988884) q[8];
rz(-1.612717193265861) q[8];
ry(-0.014735711576166821) q[9];
rz(1.9395460740258574) q[9];
ry(1.6056631377694854) q[10];
rz(2.721385330629455) q[10];
ry(-1.5614168237325334) q[11];
rz(-2.086252917257223) q[11];
ry(-1.6654112748610297) q[12];
rz(2.2515549837058946) q[12];
ry(0.0007802811598971102) q[13];
rz(2.851317568890969) q[13];
ry(3.131476646420253) q[14];
rz(2.1995445991570772) q[14];
ry(-1.5660672687110986) q[15];
rz(0.4595845632055078) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5833101794151212) q[0];
rz(-0.11615952094967033) q[0];
ry(-0.20346888151539472) q[1];
rz(2.850830279481934) q[1];
ry(0.0007582905812337271) q[2];
rz(-1.7800043892970123) q[2];
ry(3.140627269174588) q[3];
rz(-0.11379864213196633) q[3];
ry(3.1414225980612684) q[4];
rz(-1.068239688184632) q[4];
ry(-3.1380945854433313) q[5];
rz(-0.8002200476858361) q[5];
ry(-1.5706920011951595) q[6];
rz(1.570537558978921) q[6];
ry(1.5707534567470054) q[7];
rz(1.6816635282726526) q[7];
ry(-1.570392631845489) q[8];
rz(1.5112788423097117) q[8];
ry(1.5707699249698672) q[9];
rz(-1.4156527797208336) q[9];
ry(-0.0012228604158257726) q[10];
rz(0.12422896975342022) q[10];
ry(2.657478539447311e-05) q[11];
rz(2.058662613922742) q[11];
ry(0.0003660058472829003) q[12];
rz(-0.6922778394234965) q[12];
ry(3.1415639129000654) q[13];
rz(1.096714132879173) q[13];
ry(1.5700994656560165) q[14];
rz(-2.9064971208784063) q[14];
ry(-0.008908330178294115) q[15];
rz(2.6871061936399725) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.1731930749664934) q[0];
rz(-2.062365548591215) q[0];
ry(0.0010767298348373089) q[1];
rz(-1.1864591629347454) q[1];
ry(1.6013369017927301) q[2];
rz(0.7488775197803346) q[2];
ry(-1.5080213623094751) q[3];
rz(2.147135019237268) q[3];
ry(-1.573246163429605) q[4];
rz(2.770748346951673) q[4];
ry(3.136084041411896) q[5];
rz(0.6133821400776921) q[5];
ry(-1.5665464363567758) q[6];
rz(-0.37229281533769054) q[6];
ry(-3.137386157134106) q[7];
rz(2.088318956928759) q[7];
ry(0.0008412229540343011) q[8];
rz(-0.31192261087312717) q[8];
ry(-0.00026148731896658006) q[9];
rz(-1.3069211331819264) q[9];
ry(-0.014885443059228582) q[10];
rz(1.4873747649493039) q[10];
ry(-1.5711581043688936) q[11];
rz(-1.1564641120872832) q[11];
ry(-0.010594162722481926) q[12];
rz(1.2098509474767232) q[12];
ry(-3.1411068946384035) q[13];
rz(-1.3110761549065164) q[13];
ry(-0.007768828794935345) q[14];
rz(-0.6035315282352656) q[14];
ry(-0.677442537575976) q[15];
rz(1.9859752433974396) q[15];