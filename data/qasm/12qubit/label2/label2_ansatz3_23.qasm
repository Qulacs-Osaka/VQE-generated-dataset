OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.306906735185054) q[0];
rz(-2.2111479474307574) q[0];
ry(-2.6160189762221133) q[1];
rz(0.45345180824596903) q[1];
ry(-2.647838366289241) q[2];
rz(0.7317588486587505) q[2];
ry(1.6616685601034058) q[3];
rz(1.2477647276907842) q[3];
ry(2.1553616302641534) q[4];
rz(0.870657308411435) q[4];
ry(-0.4583780980193703) q[5];
rz(0.7964269258929598) q[5];
ry(3.1414553017451348) q[6];
rz(1.2489842880452864) q[6];
ry(-3.1696474995477605e-05) q[7];
rz(2.936659699802804) q[7];
ry(0.6903346601115613) q[8];
rz(1.5509196142188946) q[8];
ry(2.56354125440641) q[9];
rz(2.4657260885747005) q[9];
ry(-1.374638196989598) q[10];
rz(0.3933131843980107) q[10];
ry(0.8557468630625426) q[11];
rz(-0.6771256313743987) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.9103864545328177) q[0];
rz(0.05644662461209003) q[0];
ry(-1.556178283904915) q[1];
rz(1.4486376044647997) q[1];
ry(0.5563746696995594) q[2];
rz(1.0052192989036328) q[2];
ry(-2.45435660190884) q[3];
rz(0.5724809860640524) q[3];
ry(0.24386095098417115) q[4];
rz(-2.933931413070877) q[4];
ry(0.7651882145329516) q[5];
rz(-0.8809470979664462) q[5];
ry(-8.805907285491799e-05) q[6];
rz(1.4965345703489517) q[6];
ry(-3.1415831867044948) q[7];
rz(-0.27834471597703825) q[7];
ry(-2.459246164598365) q[8];
rz(0.07714372598118047) q[8];
ry(0.5282923524746646) q[9];
rz(-1.049659146531886) q[9];
ry(-2.4769925628264784) q[10];
rz(1.19070137104547) q[10];
ry(-2.7769494947532416) q[11];
rz(-0.7390436879813446) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.7250281732976542) q[0];
rz(-1.9174282756751049) q[0];
ry(-3.127047814656207) q[1];
rz(-1.9432680151316908) q[1];
ry(-2.6646907094132013) q[2];
rz(2.026287223421349) q[2];
ry(-0.8530929442355757) q[3];
rz(0.20269485224955416) q[3];
ry(0.6581872314391416) q[4];
rz(1.165700050307794) q[4];
ry(1.275435994708539) q[5];
rz(-1.5446964453509875) q[5];
ry(-0.00019011477778300048) q[6];
rz(2.3666659105544308) q[6];
ry(3.141567169282241) q[7];
rz(1.1111407676356961) q[7];
ry(2.6724313240748625) q[8];
rz(-1.9652660352863558) q[8];
ry(2.567489815601301) q[9];
rz(2.0790756161175743) q[9];
ry(-0.1716452631971812) q[10];
rz(1.8920351717868504) q[10];
ry(2.027183034083863) q[11];
rz(-2.3795757952549046) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.09350895062695125) q[0];
rz(0.46396149602752484) q[0];
ry(-2.679997968114696) q[1];
rz(1.5343897558086803) q[1];
ry(0.7429269198903948) q[2];
rz(1.3487801848049088) q[2];
ry(-2.3366845658033197) q[3];
rz(-2.2173622954302137) q[3];
ry(0.15813089903529765) q[4];
rz(1.2210207573292244) q[4];
ry(2.005563490868751) q[5];
rz(2.9746923217593553) q[5];
ry(-3.398366891130218e-05) q[6];
rz(1.7158762132316463) q[6];
ry(-0.00011062447002807701) q[7];
rz(-1.0744150932148302) q[7];
ry(-2.461872885750723) q[8];
rz(-2.0675832647627814) q[8];
ry(-0.035984630231403976) q[9];
rz(-0.8589119187714199) q[9];
ry(1.8768365700754028) q[10];
rz(2.5755907418365687) q[10];
ry(-0.38425813734376724) q[11];
rz(-1.4543321471839015) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.8859188514892289) q[0];
rz(2.3464985403598924) q[0];
ry(1.2602720740138764) q[1];
rz(-0.4521320700691165) q[1];
ry(-3.1344453606833795) q[2];
rz(2.302332256857044) q[2];
ry(-1.6124430307196276) q[3];
rz(-2.0351418009173257) q[3];
ry(0.7517567634008968) q[4];
rz(-2.727065770684021) q[4];
ry(3.0316926537858575) q[5];
rz(2.346659019878191) q[5];
ry(0.00021172974195149408) q[6];
rz(-0.7624289408895099) q[6];
ry(3.1415672522970377) q[7];
rz(2.257530863099808) q[7];
ry(-2.3778059242362404) q[8];
rz(-1.4456908761883973) q[8];
ry(0.33605958292427274) q[9];
rz(-1.2799786872061398) q[9];
ry(-2.3762467991433573) q[10];
rz(0.39424221682598276) q[10];
ry(-2.4886617748083872) q[11];
rz(-1.184188693454218) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.4708832958222926) q[0];
rz(-0.6929218147419745) q[0];
ry(2.863312185902975) q[1];
rz(3.069386829322332) q[1];
ry(-2.4569241358271285) q[2];
rz(1.2519571660310662) q[2];
ry(-2.2778752478192397) q[3];
rz(2.65249724005246) q[3];
ry(2.9667848740834537) q[4];
rz(1.0899712365600234) q[4];
ry(-2.412934851477578) q[5];
rz(-0.43082161528234) q[5];
ry(3.1414716512300522) q[6];
rz(-2.4013676795777332) q[6];
ry(0.0001444025811843402) q[7];
rz(-0.7114392665524166) q[7];
ry(0.6339134131260096) q[8];
rz(0.5902257000092109) q[8];
ry(2.1210718551893084) q[9];
rz(2.7606543781726165) q[9];
ry(0.41137597323726904) q[10];
rz(-2.703687246510441) q[10];
ry(2.567710806332446) q[11];
rz(-0.3155055330576486) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.4172140872353651) q[0];
rz(1.4525812532607603) q[0];
ry(-0.5425389117835113) q[1];
rz(2.2863102624142844) q[1];
ry(3.044750120924948) q[2];
rz(-0.8068111738289421) q[2];
ry(2.0017614857377213) q[3];
rz(2.9147853792136345) q[3];
ry(-0.029788680960483486) q[4];
rz(-1.655493894697362) q[4];
ry(-1.3373191136568678) q[5];
rz(0.4346472366719825) q[5];
ry(-0.00013637858613353368) q[6];
rz(-2.738697438687843) q[6];
ry(1.265477267121918) q[7];
rz(0.009161452893762245) q[7];
ry(0.6412962377314415) q[8];
rz(-0.8662742900844657) q[8];
ry(0.42874303242579875) q[9];
rz(0.5261510039808714) q[9];
ry(-2.1212305395998587) q[10];
rz(0.9426244366057925) q[10];
ry(2.0324192458244768) q[11];
rz(-2.7069693798388332) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.175323186980838) q[0];
rz(1.5381414682861987) q[0];
ry(2.9819983864383044) q[1];
rz(-1.0432142930192083) q[1];
ry(2.200159258747595) q[2];
rz(-0.07170050377141683) q[2];
ry(0.7203890474920182) q[3];
rz(2.6532037165126243) q[3];
ry(-2.4689329664480724) q[4];
rz(-2.0810141093517864) q[4];
ry(7.9854761514614e-05) q[5];
rz(0.6861120719878446) q[5];
ry(-3.140102384359217) q[6];
rz(2.7801972358070435) q[6];
ry(3.1415609014999717) q[7];
rz(-1.8567615828627106) q[7];
ry(0.6972013954316968) q[8];
rz(-1.9302641497414674) q[8];
ry(-3.0232059984409436) q[9];
rz(-3.0237761355440176) q[9];
ry(0.5309967222253205) q[10];
rz(-2.888136467918302) q[10];
ry(-3.0015042551983178) q[11];
rz(-0.7146272682663435) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.0805708009880064) q[0];
rz(1.757087667735226) q[0];
ry(1.9211230477488244) q[1];
rz(0.5385385054396762) q[1];
ry(1.3159185187704194) q[2];
rz(-1.9961336062941355) q[2];
ry(2.4652899366796226) q[3];
rz(0.8918761595270004) q[3];
ry(3.0807311053535207) q[4];
rz(1.4709504575512824) q[4];
ry(-2.366116994473505) q[5];
rz(-1.6173840464257463) q[5];
ry(0.00016331672004627897) q[6];
rz(1.008873791266963) q[6];
ry(2.2863318106936767) q[7];
rz(-2.251175820637817) q[7];
ry(-1.5061786337585827) q[8];
rz(-0.9076339031889606) q[8];
ry(0.5756840462351551) q[9];
rz(0.15008396924033285) q[9];
ry(1.5582585335549881) q[10];
rz(-0.0651640119359671) q[10];
ry(-1.6547327544188173) q[11];
rz(-0.8193611847189652) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.106989938233152) q[0];
rz(-3.1030580757857633) q[0];
ry(-1.5920856351354242) q[1];
rz(-2.128788928761442) q[1];
ry(0.9261881229501226) q[2];
rz(2.9507672684813095) q[2];
ry(-3.0451129839376962) q[3];
rz(-2.3389101357741557) q[3];
ry(2.670466399366522) q[4];
rz(0.7078358528090686) q[4];
ry(-1.0762899758518314) q[5];
rz(-0.5215895246089036) q[5];
ry(-0.0005220824113649769) q[6];
rz(-0.2663967565217886) q[6];
ry(-3.8554685001024325e-05) q[7];
rz(-2.0067811146446144) q[7];
ry(2.5927254434402) q[8];
rz(0.15109353218182128) q[8];
ry(0.07180896605317164) q[9];
rz(-1.9303601509118726) q[9];
ry(-1.5317969257491921) q[10];
rz(2.91861776733077) q[10];
ry(2.8979696138682898) q[11];
rz(1.0307562169461484) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.2433866018984014) q[0];
rz(-2.8522241760631375) q[0];
ry(1.4008575001030348) q[1];
rz(-1.058191830615507) q[1];
ry(0.05234649623861065) q[2];
rz(0.7821685174317256) q[2];
ry(0.00010791568967962137) q[3];
rz(2.389840289527069) q[3];
ry(3.1415690931053963) q[4];
rz(0.5736159287531706) q[4];
ry(-0.0001537503724108785) q[5];
rz(0.5216510043127033) q[5];
ry(3.141508571068763) q[6];
rz(2.4494899032312305) q[6];
ry(8.012548282199106e-05) q[7];
rz(1.3421993733407993) q[7];
ry(1.0312981449771286) q[8];
rz(3.0584327659989103) q[8];
ry(2.9419593700012623) q[9];
rz(-1.039513790912836) q[9];
ry(-0.3051527578139508) q[10];
rz(-2.924438283424091) q[10];
ry(-0.5164147559365049) q[11];
rz(-1.9315791204056563) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.5826605626637684) q[0];
rz(0.5851918763765401) q[0];
ry(-1.0469377742476613) q[1];
rz(-0.8557394462466867) q[1];
ry(-0.6214528960863497) q[2];
rz(0.2868564985259132) q[2];
ry(0.7378761145504984) q[3];
rz(-0.2864398700695583) q[3];
ry(3.0546607697350185) q[4];
rz(2.3034458324802514) q[4];
ry(2.065280399659671) q[5];
rz(-1.4348951797753196) q[5];
ry(-3.141372778700355) q[6];
rz(-1.8670891546700608) q[6];
ry(5.635734847775353e-05) q[7];
rz(0.8565057459623596) q[7];
ry(-2.9673176739013853) q[8];
rz(-2.7977463504581133) q[8];
ry(1.5967151732816527) q[9];
rz(1.5587769266369644) q[9];
ry(-2.5545057179908497) q[10];
rz(2.8143905922348065) q[10];
ry(0.25923824092465253) q[11];
rz(2.0963007736902153) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.293143782733008) q[0];
rz(2.8735152757888573) q[0];
ry(0.7232606887845092) q[1];
rz(0.7939348354315916) q[1];
ry(0.6718858087911388) q[2];
rz(0.8451721103238095) q[2];
ry(0.19156336773265567) q[3];
rz(2.8065848174194064) q[3];
ry(3.0724749593574003) q[4];
rz(1.8467287023225012) q[4];
ry(0.12735477596023487) q[5];
rz(2.6108927616489512) q[5];
ry(-3.141549250223936) q[6];
rz(1.4930840462605037) q[6];
ry(1.2518360192811662) q[7];
rz(-1.0244104047495306) q[7];
ry(0.008738680850592928) q[8];
rz(-1.713759429219727) q[8];
ry(1.6431341995359117) q[9];
rz(-2.493355100593852) q[9];
ry(-1.2901320706049706) q[10];
rz(2.326941027281144) q[10];
ry(-1.6469463823958324) q[11];
rz(2.705857688334326) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.7715091601067172) q[0];
rz(-2.1759046525442103) q[0];
ry(-0.5143476134539057) q[1];
rz(-2.3625309048483847) q[1];
ry(-2.1628149221525574) q[2];
rz(-0.6612822471289306) q[2];
ry(1.8372858893010984) q[3];
rz(-1.8250029151764728) q[3];
ry(1.8575981169476732) q[4];
rz(0.644524644034238) q[4];
ry(1.4663651301322228e-05) q[5];
rz(1.8050663897761776) q[5];
ry(-0.8699158661863204) q[6];
rz(0.10816368401073183) q[6];
ry(3.1415904384122912) q[7];
rz(-0.5983262808390037) q[7];
ry(1.6834942396206278) q[8];
rz(-2.9018500378510894) q[8];
ry(-1.5849423626055577) q[9];
rz(-0.5517186489852453) q[9];
ry(2.3430360990470898) q[10];
rz(1.1180751334348034) q[10];
ry(-1.2408109319280989) q[11];
rz(-1.17222319610112) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.3456412884558349) q[0];
rz(1.3504930011067229) q[0];
ry(2.6718771129120085) q[1];
rz(-2.208597379992516) q[1];
ry(3.014615657807813) q[2];
rz(-0.27901785233268056) q[2];
ry(-1.5277328896340974) q[3];
rz(-1.718898049584455) q[3];
ry(0.05671321274695007) q[4];
rz(0.3856513857738797) q[4];
ry(-0.7156557919256965) q[5];
rz(2.62007309652137) q[5];
ry(-3.1364918102860653) q[6];
rz(-3.0330873371410396) q[6];
ry(-3.1029160105998415) q[7];
rz(-0.08354485671309377) q[7];
ry(-0.0020717777375951485) q[8];
rz(1.2378185175806191) q[8];
ry(3.131859817727471) q[9];
rz(-2.4403592450609684) q[9];
ry(2.6377172193246516) q[10];
rz(2.9364539644191545) q[10];
ry(1.1120187189842126) q[11];
rz(2.69079646329865) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.9267114721298328) q[0];
rz(1.293313969282547) q[0];
ry(2.0359265437654064) q[1];
rz(1.4959144646477693) q[1];
ry(0.014067714801609734) q[2];
rz(2.9679510530783357) q[2];
ry(2.1613776999863052) q[3];
rz(-2.125985809228469) q[3];
ry(0.043282091769950196) q[4];
rz(-0.4938557215924195) q[4];
ry(-0.00020047666187483768) q[5];
rz(1.1821935781101118) q[5];
ry(-2.2334459819204113) q[6];
rz(-3.1002617274504596) q[6];
ry(3.1415023105982294) q[7];
rz(-1.1613184924200581) q[7];
ry(0.007239995708280226) q[8];
rz(-1.372210195363799) q[8];
ry(-3.1015135731411303) q[9];
rz(0.7468852142995219) q[9];
ry(3.119495048149901) q[10];
rz(0.9372625754656657) q[10];
ry(2.792698431442665) q[11];
rz(-2.8548746598288077) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.359538521665764) q[0];
rz(0.5008324675536366) q[0];
ry(1.8248638989640869) q[1];
rz(0.7063409907594763) q[1];
ry(-1.416466125838494) q[2];
rz(-2.4395109908803634) q[2];
ry(2.6391054603375417) q[3];
rz(3.0466343650264345) q[3];
ry(0.06638698701731371) q[4];
rz(0.1039336151116203) q[4];
ry(2.66588796031936) q[5];
rz(-2.2286002690975146) q[5];
ry(-3.1389344115500446) q[6];
rz(0.05280520960533063) q[6];
ry(0.9728350592168045) q[7];
rz(-1.070320152293828) q[7];
ry(-3.137893822958786) q[8];
rz(3.046815254264407) q[8];
ry(-1.4619254385267828) q[9];
rz(-0.5959109329796893) q[9];
ry(-1.0104584828922336) q[10];
rz(1.6687121128249087) q[10];
ry(1.0400784206031244) q[11];
rz(-1.6652726860933535) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-3.0801971528843235) q[0];
rz(1.3070103525206376) q[0];
ry(0.9606990396049166) q[1];
rz(-1.8517527762876707) q[1];
ry(-0.004937472502071216) q[2];
rz(2.1085256358053908) q[2];
ry(-1.8702989508773404) q[3];
rz(0.3606461729516708) q[3];
ry(0.15102232182883107) q[4];
rz(-1.735594458436883) q[4];
ry(-0.00028790101174092086) q[5];
rz(-3.111821955780027) q[5];
ry(3.076200045853695) q[6];
rz(-3.0911672504672953) q[6];
ry(3.1415205717515025) q[7];
rz(-2.282620704721272) q[7];
ry(-3.064624481208947) q[8];
rz(0.4399565696694312) q[8];
ry(3.1119089350336115) q[9];
rz(2.959242290431473) q[9];
ry(0.035715985593592556) q[10];
rz(-2.516786918852772) q[10];
ry(-1.9603819232269255) q[11];
rz(0.6527170629481411) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.42239977077523605) q[0];
rz(-1.2570541373466149) q[0];
ry(-1.3262079564886724) q[1];
rz(-1.7050313416231226) q[1];
ry(2.156227228881061) q[2];
rz(-0.16912396272873312) q[2];
ry(2.5252622468038326) q[3];
rz(-2.5508556133348095) q[3];
ry(-0.0067102247265282905) q[4];
rz(1.8793050321109366) q[4];
ry(-1.8488549708122575) q[5];
rz(-0.5467383720573258) q[5];
ry(-0.0014129104014974914) q[6];
rz(-1.179644282527063) q[6];
ry(1.1008613957681532) q[7];
rz(-0.40852933325766977) q[7];
ry(-1.646566095181087) q[8];
rz(3.036289346921679) q[8];
ry(0.6122249298138192) q[9];
rz(1.7454752730548417) q[9];
ry(0.7816345145059626) q[10];
rz(0.40553996147304966) q[10];
ry(1.1084945715097803) q[11];
rz(1.2187175238525763) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.0516917553212455) q[0];
rz(-2.565361768054602) q[0];
ry(1.5436030104985254) q[1];
rz(-2.4924696715327053) q[1];
ry(-0.001310985650835975) q[2];
rz(1.2191722303855412) q[2];
ry(0.1162631101731978) q[3];
rz(1.7495566247072805) q[3];
ry(-0.25738426623200183) q[4];
rz(-0.8783674596127086) q[4];
ry(0.04755086503766481) q[5];
rz(-1.8540167809464103) q[5];
ry(-0.00020498068273158086) q[6];
rz(1.1368486156363433) q[6];
ry(-3.140254175631677) q[7];
rz(-2.8374129910214756) q[7];
ry(-2.9557542662634364) q[8];
rz(-0.17714858357154295) q[8];
ry(0.7623650562702444) q[9];
rz(0.17398486384663592) q[9];
ry(-1.804822514899013) q[10];
rz(-2.730566198638408) q[10];
ry(-0.8398594391234601) q[11];
rz(-2.7413894057934165) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.6846959868062849) q[0];
rz(-2.543102258686513) q[0];
ry(2.82039787308608) q[1];
rz(2.619520837483407) q[1];
ry(2.273508005192838) q[2];
rz(1.8943010664236946) q[2];
ry(-1.5696510226113187) q[3];
rz(-1.5771597894960339) q[3];
ry(0.9790720582333154) q[4];
rz(2.704107966259281) q[4];
ry(-3.0478233940752317) q[5];
rz(0.6513978585681421) q[5];
ry(1.569838141992168) q[6];
rz(1.6494099364866095) q[6];
ry(0.03725172337512542) q[7];
rz(2.3180081230162886) q[7];
ry(-2.647234249420834) q[8];
rz(-0.7464279913268861) q[8];
ry(2.631389932871631) q[9];
rz(0.4976228302501778) q[9];
ry(-2.9150723985806013) q[10];
rz(-2.580498786197466) q[10];
ry(-0.8538740883808211) q[11];
rz(-0.6697142414643948) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.5536776912396317) q[0];
rz(-2.7437161949557187) q[0];
ry(-3.0173986136508706) q[1];
rz(-0.46665952375379055) q[1];
ry(-3.1414501826605368) q[2];
rz(-0.8376265497082486) q[2];
ry(1.660000306765939) q[3];
rz(-1.7570430010377558) q[3];
ry(3.1415822101081066) q[4];
rz(3.1072162485062487) q[4];
ry(-3.1413343483922103) q[5];
rz(2.553454404606287) q[5];
ry(3.141565537209253) q[6];
rz(0.07882072975315602) q[6];
ry(0.0006396686917931983) q[7];
rz(-2.4467817745156966) q[7];
ry(-1.5708657772138497) q[8];
rz(1.5708912606327772) q[8];
ry(-0.014802371393181671) q[9];
rz(0.7009537915807067) q[9];
ry(-1.3783186452773153) q[10];
rz(-2.7769660887417205) q[10];
ry(0.8830579230496086) q[11];
rz(2.3631919892410482) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.27516638110079317) q[0];
rz(0.9245370266662788) q[0];
ry(-3.0646232257448953) q[1];
rz(-3.102762598571436) q[1];
ry(-3.071388819551906) q[2];
rz(2.5455968651544043) q[2];
ry(-1.5615795377429544) q[3];
rz(-1.5636534031378169) q[3];
ry(0.8507441429621287) q[4];
rz(1.574896501131664) q[4];
ry(1.572415526987976) q[5];
rz(1.3599205152368157) q[5];
ry(0.9075754584933905) q[6];
rz(0.03983040339722308) q[6];
ry(-0.03718053538791466) q[7];
rz(-2.3535056876406917) q[7];
ry(1.5707683833739272) q[8];
rz(2.467011980670588) q[8];
ry(0.00012574930426353036) q[9];
rz(2.336161577875495) q[9];
ry(1.5708090358616653) q[10];
rz(-1.2371467411644435) q[10];
ry(1.0880254072106068) q[11];
rz(-2.3258319783303136) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.088726803808678) q[0];
rz(-2.3070335466902527) q[0];
ry(1.6260647682142526) q[1];
rz(-2.8981338299223594) q[1];
ry(0.009629001812233366) q[2];
rz(-0.8551985548548315) q[2];
ry(-0.10815931329567707) q[3];
rz(2.8394826060527762) q[3];
ry(3.141564037258321) q[4];
rz(-1.020776035370282) q[4];
ry(0.00035086004957003074) q[5];
rz(-1.8073865854893336) q[5];
ry(1.8168719306859282) q[6];
rz(-4.4189937536960426e-05) q[6];
ry(-3.1415575756915244) q[7];
rz(1.5089644219788012) q[7];
ry(-1.8825722665216007) q[8];
rz(-0.6127455891801963) q[8];
ry(-1.521805556482656) q[9];
rz(2.603073553749515) q[9];
ry(-1.0315499628319902) q[10];
rz(-1.7456234042895735) q[10];
ry(-1.5760423193631967) q[11];
rz(2.900654959348768) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.7390126395345775) q[0];
rz(2.5086897715914267) q[0];
ry(-2.8811520636641177) q[1];
rz(-1.6041014759605323) q[1];
ry(-2.8703826251638165) q[2];
rz(-0.6105773561637582) q[2];
ry(-0.16651343963444099) q[3];
rz(-1.742057944355456) q[3];
ry(3.141159567250847) q[4];
rz(2.500132989523333) q[4];
ry(-2.1140386369526447) q[5];
rz(0.2931173385095907) q[5];
ry(-1.8058695231438833) q[6];
rz(-3.1415317467246977) q[6];
ry(3.141528209015089) q[7];
rz(-3.120447318008971) q[7];
ry(3.1414051952801696) q[8];
rz(2.549279809191904) q[8];
ry(-9.479314310389954e-05) q[9];
rz(2.4451723016930793) q[9];
ry(-3.105738849029242) q[10];
rz(-1.5854316797079673) q[10];
ry(-3.140592266891157) q[11];
rz(-1.5881911100034456) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.963677653911828) q[0];
rz(-1.440154479792306) q[0];
ry(1.6026827714626413) q[1];
rz(-2.878829026844137) q[1];
ry(-0.010067382230602995) q[2];
rz(-2.5893192371904594) q[2];
ry(-3.119421971852919) q[3];
rz(1.8154016200401806) q[3];
ry(-8.146993319930397e-05) q[4];
rz(0.5128654207760583) q[4];
ry(3.1358632034181775) q[5];
rz(1.383808064959887) q[5];
ry(1.3237097072922728) q[6];
rz(-0.9498699720880301) q[6];
ry(-0.0003043291494728532) q[7];
rz(2.6818645691429968) q[7];
ry(-0.9451073487542636) q[8];
rz(0.1370311315832039) q[8];
ry(0.1466394090375136) q[9];
rz(-0.017958344536739362) q[9];
ry(-1.5989291540699355) q[10];
rz(-1.1059061291623946) q[10];
ry(0.028524294338725827) q[11];
rz(-2.805201599335839) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.509912761078103) q[0];
rz(-0.2920209171442199) q[0];
ry(0.14284185968621266) q[1];
rz(-1.9847962613728696) q[1];
ry(-0.05173480389630287) q[2];
rz(-2.4920880023839844) q[2];
ry(2.7480887878023665) q[3];
rz(-1.314092269004493) q[3];
ry(0.019234701393491704) q[4];
rz(-0.7799719435262106) q[4];
ry(1.488796444463361) q[5];
rz(0.45835466970161387) q[5];
ry(2.9623051945525187) q[6];
rz(1.756467316152365) q[6];
ry(1.692255281878971) q[7];
rz(-2.0207557721639713) q[7];
ry(-1.8110720146264898) q[8];
rz(2.7224115798369417) q[8];
ry(-0.4020652808855454) q[9];
rz(2.3942689629437583) q[9];
ry(-0.1747171900536942) q[10];
rz(0.37276402355490473) q[10];
ry(-0.2386704732755656) q[11];
rz(2.15424735714796) q[11];