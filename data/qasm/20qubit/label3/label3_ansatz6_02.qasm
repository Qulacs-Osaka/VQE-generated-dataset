OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.361944922214117) q[0];
ry(0.615213492391595) q[1];
cx q[0],q[1];
ry(-0.5658109512261128) q[0];
ry(1.0486265786661964) q[1];
cx q[0],q[1];
ry(1.4549121444998854) q[1];
ry(-2.5352168955471814) q[2];
cx q[1],q[2];
ry(2.413367697957447) q[1];
ry(1.0295068533984697) q[2];
cx q[1],q[2];
ry(0.6029218912034093) q[2];
ry(-0.14230891460343031) q[3];
cx q[2],q[3];
ry(3.1231183988043356) q[2];
ry(1.734877040743199) q[3];
cx q[2],q[3];
ry(0.7614974236343808) q[3];
ry(3.089254849795946) q[4];
cx q[3],q[4];
ry(-1.7797809286040724) q[3];
ry(1.7171841178567795) q[4];
cx q[3],q[4];
ry(-0.41820235475589745) q[4];
ry(1.5232553167969054) q[5];
cx q[4],q[5];
ry(-2.0335922894849405) q[4];
ry(-1.5742824142534415) q[5];
cx q[4],q[5];
ry(-2.3517543133202032) q[5];
ry(-2.6922732219353476) q[6];
cx q[5],q[6];
ry(1.1665624817573423) q[5];
ry(1.5542198640257965) q[6];
cx q[5],q[6];
ry(3.027446293857752) q[6];
ry(-0.4213164678995315) q[7];
cx q[6],q[7];
ry(1.1528677662238396) q[6];
ry(-1.7685958786409637) q[7];
cx q[6],q[7];
ry(1.2435260355283555) q[7];
ry(3.0076418192664) q[8];
cx q[7],q[8];
ry(-2.122689881926579) q[7];
ry(-1.64921950917611) q[8];
cx q[7],q[8];
ry(-2.6967234324297227) q[8];
ry(-3.0853842732376795) q[9];
cx q[8],q[9];
ry(0.8320068429900642) q[8];
ry(3.100710999229873) q[9];
cx q[8],q[9];
ry(-0.7126544375095643) q[9];
ry(1.223597479623334) q[10];
cx q[9],q[10];
ry(2.2116105587266897) q[9];
ry(1.9808951970010433) q[10];
cx q[9],q[10];
ry(1.7062339742059338) q[10];
ry(-0.17440711665717765) q[11];
cx q[10],q[11];
ry(0.23793595081947524) q[10];
ry(0.0018982362495929548) q[11];
cx q[10],q[11];
ry(2.2440287118457363) q[11];
ry(0.7590213787061195) q[12];
cx q[11],q[12];
ry(-0.3365896407169422) q[11];
ry(-2.8118038830799623) q[12];
cx q[11],q[12];
ry(0.29544650276089046) q[12];
ry(0.9700487352910088) q[13];
cx q[12],q[13];
ry(2.3264718700898688) q[12];
ry(-2.211350356296823) q[13];
cx q[12],q[13];
ry(2.5183704768987263) q[13];
ry(-0.9384670350292136) q[14];
cx q[13],q[14];
ry(-1.3770413706493896) q[13];
ry(1.5834579365825614) q[14];
cx q[13],q[14];
ry(-0.7846363404037877) q[14];
ry(-0.033628458937386085) q[15];
cx q[14],q[15];
ry(1.5980930494934231) q[14];
ry(1.6105175506682536) q[15];
cx q[14],q[15];
ry(-2.7709329397940836) q[15];
ry(-1.7768330966378105) q[16];
cx q[15],q[16];
ry(-1.9789275106578135) q[15];
ry(-2.9880636800114697) q[16];
cx q[15],q[16];
ry(1.4131582435936707) q[16];
ry(-3.098753511836643) q[17];
cx q[16],q[17];
ry(-2.363701250776299) q[16];
ry(1.600757038324836) q[17];
cx q[16],q[17];
ry(-0.531186817922908) q[17];
ry(-1.0742769721645746) q[18];
cx q[17],q[18];
ry(1.0934644827363178) q[17];
ry(1.605846614116273) q[18];
cx q[17],q[18];
ry(-2.784166004787189) q[18];
ry(0.6581799058285087) q[19];
cx q[18],q[19];
ry(2.2887726698575386) q[18];
ry(1.5326499905315334) q[19];
cx q[18],q[19];
ry(-2.2669487244492936) q[0];
ry(1.9136766905384253) q[1];
cx q[0],q[1];
ry(1.2179199667777616) q[0];
ry(0.5334159045205036) q[1];
cx q[0],q[1];
ry(0.6953954261458737) q[1];
ry(-0.3135627089324924) q[2];
cx q[1],q[2];
ry(-2.94920692045155) q[1];
ry(2.8180109385740155) q[2];
cx q[1],q[2];
ry(-0.15485011628802503) q[2];
ry(1.2902724629610391) q[3];
cx q[2],q[3];
ry(-0.24152806697704135) q[2];
ry(2.784670016206498) q[3];
cx q[2],q[3];
ry(-1.4812390310330201) q[3];
ry(-0.8210386881170681) q[4];
cx q[3],q[4];
ry(0.10083376839604076) q[3];
ry(-1.5891751143394073) q[4];
cx q[3],q[4];
ry(-2.4006372300346928) q[4];
ry(0.9081755643788538) q[5];
cx q[4],q[5];
ry(1.5772855032599793) q[4];
ry(0.0014177804202972766) q[5];
cx q[4],q[5];
ry(0.12643457398692082) q[5];
ry(-1.6221868576323537) q[6];
cx q[5],q[6];
ry(-0.018996289885624407) q[5];
ry(3.1180840845006257) q[6];
cx q[5],q[6];
ry(-0.034196088817113335) q[6];
ry(-1.8092780266713224) q[7];
cx q[6],q[7];
ry(1.7743428531147831) q[6];
ry(-1.338509157671985) q[7];
cx q[6],q[7];
ry(0.09308226741835829) q[7];
ry(-1.0077818367397517) q[8];
cx q[7],q[8];
ry(2.0229006386396264) q[7];
ry(1.3413779935515437) q[8];
cx q[7],q[8];
ry(-2.5791252705532153) q[8];
ry(1.5753953152129698) q[9];
cx q[8],q[9];
ry(1.0882862076103352) q[8];
ry(1.53083029953132) q[9];
cx q[8],q[9];
ry(-2.9563656432038172) q[9];
ry(-0.49430970449772976) q[10];
cx q[9],q[10];
ry(-1.7878719369081946) q[9];
ry(1.6104430852133955) q[10];
cx q[9],q[10];
ry(2.2088268181754254) q[10];
ry(1.2734776293211563) q[11];
cx q[10],q[11];
ry(-1.8791588363446319) q[10];
ry(0.012984780251793634) q[11];
cx q[10],q[11];
ry(-1.628919610131279) q[11];
ry(-1.0332958703284705) q[12];
cx q[11],q[12];
ry(-2.3852026816285674) q[11];
ry(-2.7286032346512843) q[12];
cx q[11],q[12];
ry(-1.2662975243754362) q[12];
ry(-1.484560221051342) q[13];
cx q[12],q[13];
ry(2.4415988167306124) q[12];
ry(-1.538633854323006) q[13];
cx q[12],q[13];
ry(-1.8966616879165752) q[13];
ry(-1.5544414939840001) q[14];
cx q[13],q[14];
ry(1.5546688168240161) q[13];
ry(-1.5741185495140273) q[14];
cx q[13],q[14];
ry(-1.97512285979774) q[14];
ry(-1.3800059799217115) q[15];
cx q[14],q[15];
ry(-1.0980727197816025) q[14];
ry(-1.9605103597486497) q[15];
cx q[14],q[15];
ry(-0.2963596553553902) q[15];
ry(-2.1389978949237705) q[16];
cx q[15],q[16];
ry(-1.2715226994382034) q[15];
ry(0.25836783146786535) q[16];
cx q[15],q[16];
ry(-1.1690751281222644) q[16];
ry(1.5976119839159038) q[17];
cx q[16],q[17];
ry(-2.0428736732904076) q[16];
ry(-2.4425921504815613) q[17];
cx q[16],q[17];
ry(0.02128075595445031) q[17];
ry(-1.5731664105406453) q[18];
cx q[17],q[18];
ry(-1.706855516605694) q[17];
ry(1.5778781410859315) q[18];
cx q[17],q[18];
ry(-1.3542530547651195) q[18];
ry(2.722119243637195) q[19];
cx q[18],q[19];
ry(2.4620380973167064) q[18];
ry(-0.025667759110345037) q[19];
cx q[18],q[19];
ry(-0.37078159069690353) q[0];
ry(-0.3005237798235365) q[1];
cx q[0],q[1];
ry(-0.7315408526576794) q[0];
ry(-0.12589826783168973) q[1];
cx q[0],q[1];
ry(-2.587439460241865) q[1];
ry(2.3284117436994105) q[2];
cx q[1],q[2];
ry(-2.7967309550725874) q[1];
ry(2.9839990431177124) q[2];
cx q[1],q[2];
ry(2.057843205859996) q[2];
ry(1.6021214203841236) q[3];
cx q[2],q[3];
ry(-0.37039520660319386) q[2];
ry(1.613665472912884) q[3];
cx q[2],q[3];
ry(-1.7966286892712242) q[3];
ry(-3.1256332992554734) q[4];
cx q[3],q[4];
ry(-1.7670813202800901) q[3];
ry(-3.1375318171823796) q[4];
cx q[3],q[4];
ry(-1.5301800736270488) q[4];
ry(1.7265799880025756) q[5];
cx q[4],q[5];
ry(1.3136127339546162) q[4];
ry(-0.21074113519144339) q[5];
cx q[4],q[5];
ry(-0.05683528219343614) q[5];
ry(2.7893044321386595) q[6];
cx q[5],q[6];
ry(1.570511882202271) q[5];
ry(-1.572248886610958) q[6];
cx q[5],q[6];
ry(0.009492631954342913) q[6];
ry(-1.142717563671342) q[7];
cx q[6],q[7];
ry(0.005267389267165967) q[6];
ry(-1.6505879259288214) q[7];
cx q[6],q[7];
ry(-1.0263727274865146) q[7];
ry(-1.5713505727824488) q[8];
cx q[7],q[8];
ry(1.1340525231389833) q[7];
ry(-1.500185411000695) q[8];
cx q[7],q[8];
ry(1.5710511356901375) q[8];
ry(-0.08945945810620409) q[9];
cx q[8],q[9];
ry(-1.470860602000256) q[8];
ry(-1.4772208686150288) q[9];
cx q[8],q[9];
ry(0.9351314212397108) q[9];
ry(0.9258931927004417) q[10];
cx q[9],q[10];
ry(-1.5843796724342658) q[9];
ry(-1.7556363967750155) q[10];
cx q[9],q[10];
ry(1.4609832464700883) q[10];
ry(2.9982195122287183) q[11];
cx q[10],q[11];
ry(1.5324731106768341) q[10];
ry(1.5914213663064236) q[11];
cx q[10],q[11];
ry(1.384339492096763) q[11];
ry(-0.9857202117998111) q[12];
cx q[11],q[12];
ry(-3.1104870429826375) q[11];
ry(-1.676976166875349) q[12];
cx q[11],q[12];
ry(-2.6928941071674313) q[12];
ry(1.576105269603334) q[13];
cx q[12],q[13];
ry(-1.377068710412333) q[12];
ry(1.301036094640982) q[13];
cx q[12],q[13];
ry(-1.628295112038825) q[13];
ry(-1.5820970629413837) q[14];
cx q[13],q[14];
ry(-1.6219626170047357) q[13];
ry(3.0773198301411067) q[14];
cx q[13],q[14];
ry(0.4039001839340264) q[14];
ry(-1.536846913390635) q[15];
cx q[14],q[15];
ry(1.5902509627194918) q[14];
ry(-3.134231469454021) q[15];
cx q[14],q[15];
ry(-0.5635073365959012) q[15];
ry(-0.046657038457170716) q[16];
cx q[15],q[16];
ry(1.1887511549828105) q[15];
ry(3.1370191277233004) q[16];
cx q[15],q[16];
ry(3.0696948727690634) q[16];
ry(1.894790395605267) q[17];
cx q[16],q[17];
ry(1.5781173924382124) q[16];
ry(-1.5593312774363213) q[17];
cx q[16],q[17];
ry(3.1323742527521468) q[17];
ry(-1.645107447775163) q[18];
cx q[17],q[18];
ry(3.0852936868686687) q[17];
ry(1.5640598227033058) q[18];
cx q[17],q[18];
ry(0.18717245863889673) q[18];
ry(2.145254223248778) q[19];
cx q[18],q[19];
ry(0.28557779143729967) q[18];
ry(-0.548008698529447) q[19];
cx q[18],q[19];
ry(2.7041661815308595) q[0];
ry(-0.5319117035757499) q[1];
cx q[0],q[1];
ry(-0.7598273126874542) q[0];
ry(1.4449541075418573) q[1];
cx q[0],q[1];
ry(-0.5416591670858804) q[1];
ry(-2.1266829691427946) q[2];
cx q[1],q[2];
ry(0.036217978255492375) q[1];
ry(1.5695857406161178) q[2];
cx q[1],q[2];
ry(-2.586821579886424) q[2];
ry(-0.2552983616231509) q[3];
cx q[2],q[3];
ry(1.5721691801423303) q[2];
ry(-1.5695200908740654) q[3];
cx q[2],q[3];
ry(0.025542639571257908) q[3];
ry(3.0990729978876734) q[4];
cx q[3],q[4];
ry(-3.061926953504604) q[3];
ry(2.8176060099270996) q[4];
cx q[3],q[4];
ry(-1.569979484496475) q[4];
ry(-3.112440807966671) q[5];
cx q[4],q[5];
ry(-2.6854015371224116) q[4];
ry(-2.6532541164263934) q[5];
cx q[4],q[5];
ry(0.5069167088310657) q[5];
ry(0.020817439063626555) q[6];
cx q[5],q[6];
ry(-1.5723267231964686) q[5];
ry(1.5653673106690853) q[6];
cx q[5],q[6];
ry(-2.2588857896227887) q[6];
ry(-1.837642824126) q[7];
cx q[6],q[7];
ry(-3.1181182213647176) q[6];
ry(0.009063977759142361) q[7];
cx q[6],q[7];
ry(1.3070062769015847) q[7];
ry(2.325134644614079) q[8];
cx q[7],q[8];
ry(-1.7540306458345754) q[7];
ry(-0.8362585425729918) q[8];
cx q[7],q[8];
ry(2.9969465268278874) q[8];
ry(1.5584102566407059) q[9];
cx q[8],q[9];
ry(-0.07823452473519144) q[8];
ry(-1.6634335801117368) q[9];
cx q[8],q[9];
ry(-2.986227687873692) q[9];
ry(0.005616476325330132) q[10];
cx q[9],q[10];
ry(-1.5643242774746025) q[9];
ry(0.4405564527983264) q[10];
cx q[9],q[10];
ry(-2.2574000019012823) q[10];
ry(-3.0042671815988875) q[11];
cx q[10],q[11];
ry(-0.0011099852091108928) q[10];
ry(3.1383272452198194) q[11];
cx q[10],q[11];
ry(1.477467394032416) q[11];
ry(0.04644303053111454) q[12];
cx q[11],q[12];
ry(1.6046004927330326) q[11];
ry(-0.7154377056156226) q[12];
cx q[11],q[12];
ry(-0.0007506143236604146) q[12];
ry(-1.315254939082899) q[13];
cx q[12],q[13];
ry(0.017099820065470794) q[12];
ry(-1.652194916772462) q[13];
cx q[12],q[13];
ry(-0.405252980853738) q[13];
ry(-2.694830767607302) q[14];
cx q[13],q[14];
ry(-2.9443457271458895) q[13];
ry(-1.5862823291877577) q[14];
cx q[13],q[14];
ry(-2.846656234263106) q[14];
ry(1.063520167640148) q[15];
cx q[14],q[15];
ry(1.5715962349838701) q[14];
ry(-3.126549815114204) q[15];
cx q[14],q[15];
ry(-1.8086914924022457) q[15];
ry(1.9218493250731752) q[16];
cx q[15],q[16];
ry(-3.104398981264055) q[15];
ry(-3.1394129590627236) q[16];
cx q[15],q[16];
ry(0.668416831617253) q[16];
ry(-1.7542565623734063) q[17];
cx q[16],q[17];
ry(1.8168505828613533) q[16];
ry(-1.531588215204852) q[17];
cx q[16],q[17];
ry(0.1327078607728145) q[17];
ry(1.5566622612028773) q[18];
cx q[17],q[18];
ry(-1.5936617305005087) q[17];
ry(1.5613236031384934) q[18];
cx q[17],q[18];
ry(-0.03312465999275378) q[18];
ry(-0.5825322395960664) q[19];
cx q[18],q[19];
ry(-3.1012027848041006) q[18];
ry(0.005918331973282752) q[19];
cx q[18],q[19];
ry(-0.7463827703649991) q[0];
ry(1.5626315727868447) q[1];
cx q[0],q[1];
ry(2.237717687071235) q[0];
ry(-1.5843673305729835) q[1];
cx q[0],q[1];
ry(3.129191257273428) q[1];
ry(0.2978655495460537) q[2];
cx q[1],q[2];
ry(-1.6315357473600969) q[1];
ry(-0.1899949354756008) q[2];
cx q[1],q[2];
ry(-3.0998880838166016) q[2];
ry(-0.09398207789243873) q[3];
cx q[2],q[3];
ry(1.5678507154711518) q[2];
ry(1.5744026407352587) q[3];
cx q[2],q[3];
ry(3.0996431376834352) q[3];
ry(0.05115147802697795) q[4];
cx q[3],q[4];
ry(-3.108321532757872) q[3];
ry(-3.099205745395711) q[4];
cx q[3],q[4];
ry(1.649221282618164) q[4];
ry(-3.137139066193016) q[5];
cx q[4],q[5];
ry(-1.5365484187188232) q[4];
ry(3.077490145958647) q[5];
cx q[4],q[5];
ry(1.564431581770957) q[5];
ry(-2.6331154079525683) q[6];
cx q[5],q[6];
ry(-0.025179964307462832) q[5];
ry(2.6514360607455534) q[6];
cx q[5],q[6];
ry(1.1594529002325569) q[6];
ry(-0.8697974297277522) q[7];
cx q[6],q[7];
ry(0.01854514120152989) q[6];
ry(0.0010905586584479818) q[7];
cx q[6],q[7];
ry(-2.294822263179486) q[7];
ry(-1.4832832173451511) q[8];
cx q[7],q[8];
ry(0.7124931492310589) q[7];
ry(-1.5479296449500926) q[8];
cx q[7],q[8];
ry(2.709463744890807) q[8];
ry(-0.027494320876185648) q[9];
cx q[8],q[9];
ry(-1.5669335366955455) q[8];
ry(1.5820039946754074) q[9];
cx q[8],q[9];
ry(-0.12123586668023556) q[9];
ry(1.5112700340401268) q[10];
cx q[9],q[10];
ry(-0.01822321833319762) q[9];
ry(-0.024700466611360383) q[10];
cx q[9],q[10];
ry(-1.1565268355174236) q[10];
ry(-3.1329352786333127) q[11];
cx q[10],q[11];
ry(0.00017249440712507894) q[10];
ry(0.017391384459368248) q[11];
cx q[10],q[11];
ry(0.007181294360451673) q[11];
ry(-2.2679850730103963) q[12];
cx q[11],q[12];
ry(-3.0775727701620545) q[11];
ry(1.4973195194115734) q[12];
cx q[11],q[12];
ry(-2.247125131369497) q[12];
ry(1.4713678098236234) q[13];
cx q[12],q[13];
ry(1.5174129411429957) q[12];
ry(1.5934271377461113) q[13];
cx q[12],q[13];
ry(-3.1101797528498953) q[13];
ry(0.18167671048813808) q[14];
cx q[13],q[14];
ry(-1.5443356089199487) q[13];
ry(1.5669485558069765) q[14];
cx q[13],q[14];
ry(0.0037921016809834285) q[14];
ry(1.3443277421681252) q[15];
cx q[14],q[15];
ry(-1.6082254618037286) q[14];
ry(-1.5536320002040975) q[15];
cx q[14],q[15];
ry(-0.0006844355238282419) q[15];
ry(-1.5442113440283656) q[16];
cx q[15],q[16];
ry(1.578257671633387) q[15];
ry(-0.5258501417248675) q[16];
cx q[15],q[16];
ry(0.005398504825609329) q[16];
ry(-1.6111201018041896) q[17];
cx q[16],q[17];
ry(-1.5058949004244786) q[16];
ry(-1.5529533015856494) q[17];
cx q[16],q[17];
ry(-0.025239468264198785) q[17];
ry(-3.1328792089941806) q[18];
cx q[17],q[18];
ry(1.5621813875927595) q[17];
ry(-1.4489647802134253) q[18];
cx q[17],q[18];
ry(0.00011101201813437515) q[18];
ry(1.4119018740659284) q[19];
cx q[18],q[19];
ry(-1.5331345757615074) q[18];
ry(-1.5806114432980758) q[19];
cx q[18],q[19];
ry(-1.4388689258216092) q[0];
ry(1.6741873001037424) q[1];
ry(-1.4539899273141925) q[2];
ry(0.06058675042654052) q[3];
ry(-2.988121135413161) q[4];
ry(0.07974305000225154) q[5];
ry(0.7182340235500214) q[6];
ry(-2.923402939928953) q[7];
ry(1.6860752930283205) q[8];
ry(-2.7614453815303683) q[9];
ry(-0.610399238614507) q[10];
ry(-1.0784722394549722) q[11];
ry(0.5350557903618753) q[12];
ry(-0.25109617410287033) q[13];
ry(0.2220610925102422) q[14];
ry(-2.547660518473035) q[15];
ry(-2.9037787728433315) q[16];
ry(-2.5450987437138397) q[17];
ry(-2.8959472676342775) q[18];
ry(2.168915076804759) q[19];