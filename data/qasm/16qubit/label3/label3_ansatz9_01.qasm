OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.5723690134026722) q[0];
ry(-0.28630115741774187) q[1];
cx q[0],q[1];
ry(0.9687094455449143) q[0];
ry(-0.03313383205018994) q[1];
cx q[0],q[1];
ry(-2.040147749946084) q[2];
ry(-1.633627081240042) q[3];
cx q[2],q[3];
ry(-2.708474535551389) q[2];
ry(0.5393715676480255) q[3];
cx q[2],q[3];
ry(-1.7775450474902506) q[4];
ry(-2.7185700088961053) q[5];
cx q[4],q[5];
ry(-0.9323296121005872) q[4];
ry(1.403749503301974) q[5];
cx q[4],q[5];
ry(-1.6119525132564745) q[6];
ry(-1.6937392357413394) q[7];
cx q[6],q[7];
ry(-1.2558882478859656) q[6];
ry(2.6387335594162513) q[7];
cx q[6],q[7];
ry(1.7580554970990352) q[8];
ry(-3.072662775715316) q[9];
cx q[8],q[9];
ry(-2.8633871694998265) q[8];
ry(0.6804701419555315) q[9];
cx q[8],q[9];
ry(-0.396383487853436) q[10];
ry(-0.44622380747222135) q[11];
cx q[10],q[11];
ry(1.5897087270689285) q[10];
ry(1.861109144680502) q[11];
cx q[10],q[11];
ry(-2.0722768610051774) q[12];
ry(2.3170850995164556) q[13];
cx q[12],q[13];
ry(1.7541303346422206) q[12];
ry(1.2583970680249423) q[13];
cx q[12],q[13];
ry(-0.7349115950486862) q[14];
ry(-1.8463080031954184) q[15];
cx q[14],q[15];
ry(0.025700447634080113) q[14];
ry(0.1623474833580868) q[15];
cx q[14],q[15];
ry(-2.3789983582184098) q[0];
ry(-2.302007964016375) q[2];
cx q[0],q[2];
ry(-2.8067244536834535) q[0];
ry(-1.8808122542176455) q[2];
cx q[0],q[2];
ry(1.773460561680018) q[2];
ry(-2.6653613334694803) q[4];
cx q[2],q[4];
ry(2.2014517562186926) q[2];
ry(-2.4771840471292017) q[4];
cx q[2],q[4];
ry(1.2698669330975907) q[4];
ry(-0.0370507593306106) q[6];
cx q[4],q[6];
ry(-2.2251309575487612) q[4];
ry(-1.4274640990850047) q[6];
cx q[4],q[6];
ry(0.6134101947622268) q[6];
ry(-0.8134538891436806) q[8];
cx q[6],q[8];
ry(-3.140174893228582) q[6];
ry(0.0009506877617824624) q[8];
cx q[6],q[8];
ry(2.5325542901579414) q[8];
ry(-0.2116534695221267) q[10];
cx q[8],q[10];
ry(1.4557290454729044) q[8];
ry(-3.107608901434275) q[10];
cx q[8],q[10];
ry(-2.385672456202191) q[10];
ry(-0.09553223910423903) q[12];
cx q[10],q[12];
ry(1.008182565331766) q[10];
ry(-2.553714612275577) q[12];
cx q[10],q[12];
ry(-1.6732951099578455) q[12];
ry(1.3878072170107503) q[14];
cx q[12],q[14];
ry(-0.38363787257656234) q[12];
ry(0.521597022070031) q[14];
cx q[12],q[14];
ry(0.28896833561152047) q[1];
ry(-0.8519920242811585) q[3];
cx q[1],q[3];
ry(0.24946401663465387) q[1];
ry(0.5403742487002434) q[3];
cx q[1],q[3];
ry(0.6354077002556124) q[3];
ry(0.5334737532572804) q[5];
cx q[3],q[5];
ry(0.3950182287813363) q[3];
ry(2.3246044386287092) q[5];
cx q[3],q[5];
ry(-1.396896779503912) q[5];
ry(2.97985181072844) q[7];
cx q[5],q[7];
ry(0.009798908926966075) q[5];
ry(0.023683043031056172) q[7];
cx q[5],q[7];
ry(-2.315329209348233) q[7];
ry(1.2441042525441466) q[9];
cx q[7],q[9];
ry(0.043307354461244785) q[7];
ry(0.040928957959027734) q[9];
cx q[7],q[9];
ry(2.1188607137962006) q[9];
ry(-1.7585081832280105) q[11];
cx q[9],q[11];
ry(2.9849796480817776) q[9];
ry(1.9657636131100755) q[11];
cx q[9],q[11];
ry(-1.5780133220958588) q[11];
ry(2.688756522151688) q[13];
cx q[11],q[13];
ry(2.8139292508172846) q[11];
ry(-3.0782943480817044) q[13];
cx q[11],q[13];
ry(-1.9891663196062652) q[13];
ry(0.1457929043522999) q[15];
cx q[13],q[15];
ry(-2.633218776876744) q[13];
ry(0.0621274678457106) q[15];
cx q[13],q[15];
ry(-3.1387135205582593) q[0];
ry(-2.7381784232422235) q[3];
cx q[0],q[3];
ry(-1.6746941789742316) q[0];
ry(1.6025310430368298) q[3];
cx q[0],q[3];
ry(0.35824978733185175) q[1];
ry(1.9418887463921788) q[2];
cx q[1],q[2];
ry(-0.19190346236186206) q[1];
ry(1.6311675791602704) q[2];
cx q[1],q[2];
ry(-0.691218818800917) q[2];
ry(-0.8647676803981977) q[5];
cx q[2],q[5];
ry(0.22247936748963415) q[2];
ry(-0.415812518331095) q[5];
cx q[2],q[5];
ry(-0.10684210560097185) q[3];
ry(-1.9967328780131655) q[4];
cx q[3],q[4];
ry(0.31464875438791307) q[3];
ry(0.005726783341993169) q[4];
cx q[3],q[4];
ry(0.6521861577970453) q[4];
ry(1.0550894837242515) q[7];
cx q[4],q[7];
ry(0.040770861737531305) q[4];
ry(0.08780721162343497) q[7];
cx q[4],q[7];
ry(0.3876322812696573) q[5];
ry(1.9731105338703117) q[6];
cx q[5],q[6];
ry(-2.172910703300955) q[5];
ry(-2.741021616469508) q[6];
cx q[5],q[6];
ry(-1.042358116538245) q[6];
ry(0.43160786510997873) q[9];
cx q[6],q[9];
ry(-3.1180982766468364) q[6];
ry(-0.017197017133846693) q[9];
cx q[6],q[9];
ry(-0.5022476556166557) q[7];
ry(-1.4885853300591745) q[8];
cx q[7],q[8];
ry(-0.031373172863672354) q[7];
ry(-3.1059304281685485) q[8];
cx q[7],q[8];
ry(1.8058731153990593) q[8];
ry(-0.16852049062450278) q[11];
cx q[8],q[11];
ry(-1.8307724319971683) q[8];
ry(-1.0619270902692195) q[11];
cx q[8],q[11];
ry(1.8786179911810592) q[9];
ry(1.48801835042077) q[10];
cx q[9],q[10];
ry(-1.5486296119993135) q[9];
ry(-1.7032597375049843) q[10];
cx q[9],q[10];
ry(1.2489563000992794) q[10];
ry(1.549283864690511) q[13];
cx q[10],q[13];
ry(0.8622716436167011) q[10];
ry(-1.4615117015413777) q[13];
cx q[10],q[13];
ry(-1.9221564580760055) q[11];
ry(2.781504980339117) q[12];
cx q[11],q[12];
ry(2.9378377298679568) q[11];
ry(2.7001965949195306) q[12];
cx q[11],q[12];
ry(2.782260529832754) q[12];
ry(1.4934453720321574) q[15];
cx q[12],q[15];
ry(2.287031077766117) q[12];
ry(1.1181321062558363) q[15];
cx q[12],q[15];
ry(-0.008069812081505557) q[13];
ry(-0.1104936643753458) q[14];
cx q[13],q[14];
ry(-3.0800941154260393) q[13];
ry(-0.019457776627803014) q[14];
cx q[13],q[14];
ry(-0.23520296287874348) q[0];
ry(-2.512518383912869) q[1];
cx q[0],q[1];
ry(1.6884621714174024) q[0];
ry(-2.516541644906641) q[1];
cx q[0],q[1];
ry(3.119444939966657) q[2];
ry(3.058277254736974) q[3];
cx q[2],q[3];
ry(3.0851510791993806) q[2];
ry(2.3112218093158217) q[3];
cx q[2],q[3];
ry(0.5919352238573696) q[4];
ry(-0.23026149595080536) q[5];
cx q[4],q[5];
ry(1.376301674113535) q[4];
ry(1.62598298936625) q[5];
cx q[4],q[5];
ry(-3.124025594721552) q[6];
ry(-0.8499587085219844) q[7];
cx q[6],q[7];
ry(3.1362438077353976) q[6];
ry(-0.04916930105347607) q[7];
cx q[6],q[7];
ry(-0.044258146918251036) q[8];
ry(-1.5932703810552566) q[9];
cx q[8],q[9];
ry(-1.4878827333233797) q[8];
ry(-3.05510898279078) q[9];
cx q[8],q[9];
ry(-3.129231037808569) q[10];
ry(-0.9097674606681311) q[11];
cx q[10],q[11];
ry(-1.5169680658562175) q[10];
ry(1.523438773696222) q[11];
cx q[10],q[11];
ry(1.5139951431626528) q[12];
ry(-1.1605307140735224) q[13];
cx q[12],q[13];
ry(-3.0845747663186387) q[12];
ry(2.9163904573523545) q[13];
cx q[12],q[13];
ry(-2.6279450470667034) q[14];
ry(2.005891560731427) q[15];
cx q[14],q[15];
ry(-2.9243879514653286) q[14];
ry(0.966024105593462) q[15];
cx q[14],q[15];
ry(2.8526025081151962) q[0];
ry(-0.2230355114270255) q[2];
cx q[0],q[2];
ry(-1.5720137696960175) q[0];
ry(-1.0010217487934854) q[2];
cx q[0],q[2];
ry(0.5154496332886639) q[2];
ry(-0.6975430910746422) q[4];
cx q[2],q[4];
ry(-0.2945326094211829) q[2];
ry(-2.7880178704595004) q[4];
cx q[2],q[4];
ry(-1.3083353749006772) q[4];
ry(-1.5509398416621716) q[6];
cx q[4],q[6];
ry(0.7636079732169712) q[4];
ry(-0.6597479394417682) q[6];
cx q[4],q[6];
ry(1.9714608677899956) q[6];
ry(1.9916656156660881) q[8];
cx q[6],q[8];
ry(-0.0020119130759542136) q[6];
ry(-0.08037174584623365) q[8];
cx q[6],q[8];
ry(1.7828401984036841) q[8];
ry(-3.086562913470999) q[10];
cx q[8],q[10];
ry(-2.7725081111829626) q[8];
ry(3.0862352488302913) q[10];
cx q[8],q[10];
ry(-2.5290418625680204) q[10];
ry(-2.161591112853983) q[12];
cx q[10],q[12];
ry(2.940321990882258) q[10];
ry(-2.8499414955081863) q[12];
cx q[10],q[12];
ry(-2.5767682649471095) q[12];
ry(1.254560696539408) q[14];
cx q[12],q[14];
ry(1.455246299701985) q[12];
ry(2.675835406265225) q[14];
cx q[12],q[14];
ry(2.4961020770302085) q[1];
ry(2.3801104001568056) q[3];
cx q[1],q[3];
ry(-1.8977278125601282) q[1];
ry(-1.0935148694359704) q[3];
cx q[1],q[3];
ry(0.6295589396808448) q[3];
ry(-1.02864539874961) q[5];
cx q[3],q[5];
ry(0.3441378214265747) q[3];
ry(2.7093791953277333) q[5];
cx q[3],q[5];
ry(1.0526338936484092) q[5];
ry(0.2067645876365031) q[7];
cx q[5],q[7];
ry(0.04572968569061312) q[5];
ry(-0.06424026060972317) q[7];
cx q[5],q[7];
ry(1.3179711628933537) q[7];
ry(2.6898566690268595) q[9];
cx q[7],q[9];
ry(-0.001219456511351482) q[7];
ry(-0.5556562635655329) q[9];
cx q[7],q[9];
ry(-0.41918763069599946) q[9];
ry(-0.09239181113843653) q[11];
cx q[9],q[11];
ry(-3.084625704859085) q[9];
ry(-3.119848262152061) q[11];
cx q[9],q[11];
ry(-1.6235999041636195) q[11];
ry(0.747421650312413) q[13];
cx q[11],q[13];
ry(1.9158041936024137) q[11];
ry(-1.3757976384322328) q[13];
cx q[11],q[13];
ry(-1.9783653424172059) q[13];
ry(0.24171031005399346) q[15];
cx q[13],q[15];
ry(-3.043071050900198) q[13];
ry(-0.0976620781771711) q[15];
cx q[13],q[15];
ry(2.8011940595592892) q[0];
ry(1.6966795894217825) q[3];
cx q[0],q[3];
ry(1.2205070423517483) q[0];
ry(-0.8116506328333137) q[3];
cx q[0],q[3];
ry(-0.39909286518123466) q[1];
ry(1.4966792755122231) q[2];
cx q[1],q[2];
ry(2.2838051676622535) q[1];
ry(3.057820986487063) q[2];
cx q[1],q[2];
ry(0.6051252608860567) q[2];
ry(-2.5817953609871394) q[5];
cx q[2],q[5];
ry(0.27860288192558236) q[2];
ry(-1.4079435199717425) q[5];
cx q[2],q[5];
ry(-0.4834423590836963) q[3];
ry(-0.11952782617506123) q[4];
cx q[3],q[4];
ry(-3.099499497494989) q[3];
ry(0.06683075094763247) q[4];
cx q[3],q[4];
ry(-1.527154841775888) q[4];
ry(1.5194451995810752) q[7];
cx q[4],q[7];
ry(-3.124364733276615) q[4];
ry(0.03317350548388389) q[7];
cx q[4],q[7];
ry(2.673367563211894) q[5];
ry(-1.3236205799052243) q[6];
cx q[5],q[6];
ry(-0.5060729791730537) q[5];
ry(2.8930626049007233) q[6];
cx q[5],q[6];
ry(-1.5994745879395056) q[6];
ry(-1.6235378753607559) q[9];
cx q[6],q[9];
ry(3.140718027938162) q[6];
ry(0.0580646693138363) q[9];
cx q[6],q[9];
ry(-2.035757003602809) q[7];
ry(-0.8092362215284385) q[8];
cx q[7],q[8];
ry(-0.006875862846324868) q[7];
ry(0.02819971802544315) q[8];
cx q[7],q[8];
ry(2.0029969151791205) q[8];
ry(2.7185831860794245) q[11];
cx q[8],q[11];
ry(-1.1997481423844052) q[8];
ry(-3.119429200393657) q[11];
cx q[8],q[11];
ry(2.948288450519294) q[9];
ry(-0.8447947685552374) q[10];
cx q[9],q[10];
ry(3.018146212422158) q[9];
ry(-3.1307400664547904) q[10];
cx q[9],q[10];
ry(-0.641685141188856) q[10];
ry(-0.4989632406539464) q[13];
cx q[10],q[13];
ry(-1.5825133313884963) q[10];
ry(-2.430993999488858) q[13];
cx q[10],q[13];
ry(-1.821860376574584) q[11];
ry(2.8501417873599517) q[12];
cx q[11],q[12];
ry(0.03817833581178309) q[11];
ry(-0.1224418676754173) q[12];
cx q[11],q[12];
ry(-0.9733402712501594) q[12];
ry(-2.9866523018778275) q[15];
cx q[12],q[15];
ry(0.7546187893485863) q[12];
ry(-0.48851293670595164) q[15];
cx q[12],q[15];
ry(1.333827222935339) q[13];
ry(1.059040854409924) q[14];
cx q[13],q[14];
ry(-2.13978619095224) q[13];
ry(-2.964337064903961) q[14];
cx q[13],q[14];
ry(-1.5097690149527812) q[0];
ry(-0.2079525179072936) q[1];
cx q[0],q[1];
ry(-1.8440602105136463) q[0];
ry(-2.872510773758375) q[1];
cx q[0],q[1];
ry(2.8580714366407727) q[2];
ry(-1.2233849682254538) q[3];
cx q[2],q[3];
ry(0.3870728912327257) q[2];
ry(2.365671884810401) q[3];
cx q[2],q[3];
ry(-0.9810002659692609) q[4];
ry(-1.295513486772528) q[5];
cx q[4],q[5];
ry(-1.02620550376313) q[4];
ry(-1.5101090281807572) q[5];
cx q[4],q[5];
ry(-1.4192521267454445) q[6];
ry(2.0732522463201013) q[7];
cx q[6],q[7];
ry(-3.085411611742625) q[6];
ry(-0.02397065157149592) q[7];
cx q[6],q[7];
ry(2.4675809195069704) q[8];
ry(0.3733035455216773) q[9];
cx q[8],q[9];
ry(0.006767429927522663) q[8];
ry(-1.5763831723804147) q[9];
cx q[8],q[9];
ry(2.072569936216242) q[10];
ry(1.213270374227191) q[11];
cx q[10],q[11];
ry(0.42079167495750625) q[10];
ry(2.8075372364533124) q[11];
cx q[10],q[11];
ry(-3.1187621723439145) q[12];
ry(1.3304277287658814) q[13];
cx q[12],q[13];
ry(-0.10914905660191998) q[12];
ry(1.6013029764440752) q[13];
cx q[12],q[13];
ry(3.1090421920536886) q[14];
ry(-1.99245506887125) q[15];
cx q[14],q[15];
ry(-2.2754490832438528) q[14];
ry(0.20178731812232176) q[15];
cx q[14],q[15];
ry(-0.26222872802182257) q[0];
ry(-1.691341612770347) q[2];
cx q[0],q[2];
ry(1.175597665403913) q[0];
ry(-0.27930559160844576) q[2];
cx q[0],q[2];
ry(1.6360674288276407) q[2];
ry(0.4946228169945641) q[4];
cx q[2],q[4];
ry(0.09822816188139025) q[2];
ry(-2.805918444633778) q[4];
cx q[2],q[4];
ry(-0.10193553159016157) q[4];
ry(-0.6495833527297128) q[6];
cx q[4],q[6];
ry(0.4323650953792769) q[4];
ry(0.6183199888419946) q[6];
cx q[4],q[6];
ry(-2.2869105841487296) q[6];
ry(-0.8918069428203124) q[8];
cx q[6],q[8];
ry(0.05184459172444923) q[6];
ry(2.8540473154826485) q[8];
cx q[6],q[8];
ry(0.8768725431812869) q[8];
ry(-1.8981884638397597) q[10];
cx q[8],q[10];
ry(-0.038902129546576354) q[8];
ry(-1.4759112749167356) q[10];
cx q[8],q[10];
ry(-1.9337202478044186) q[10];
ry(-2.1347608529735993) q[12];
cx q[10],q[12];
ry(0.19469541264029255) q[10];
ry(0.013828307854546473) q[12];
cx q[10],q[12];
ry(1.6823976255723627) q[12];
ry(2.3641832821397744) q[14];
cx q[12],q[14];
ry(3.0323429684669296) q[12];
ry(-0.217217849233534) q[14];
cx q[12],q[14];
ry(2.835719366385912) q[1];
ry(-1.6402472380644673) q[3];
cx q[1],q[3];
ry(2.8740005518273484) q[1];
ry(-1.9763624598196226) q[3];
cx q[1],q[3];
ry(0.18729114507898367) q[3];
ry(-3.11319224131591) q[5];
cx q[3],q[5];
ry(-2.998989043091153) q[3];
ry(2.928983126643858) q[5];
cx q[3],q[5];
ry(-2.0082508105474557) q[5];
ry(-2.9431033434707445) q[7];
cx q[5],q[7];
ry(3.1013625249908796) q[5];
ry(-2.989729220935633) q[7];
cx q[5],q[7];
ry(1.7145157122678611) q[7];
ry(0.479864037660501) q[9];
cx q[7],q[9];
ry(3.1024052204922086) q[7];
ry(-2.3470137625313603) q[9];
cx q[7],q[9];
ry(1.571964337149634) q[9];
ry(2.7766718829939223) q[11];
cx q[9],q[11];
ry(1.951053479395362) q[9];
ry(1.6892259423916727) q[11];
cx q[9],q[11];
ry(2.213930057174607) q[11];
ry(0.37716974450923996) q[13];
cx q[11],q[13];
ry(-0.014745712529744765) q[11];
ry(0.3958409887615115) q[13];
cx q[11],q[13];
ry(-1.7997673369129936) q[13];
ry(-0.46785024484962784) q[15];
cx q[13],q[15];
ry(-0.7278790805481006) q[13];
ry(-1.9344819328311607) q[15];
cx q[13],q[15];
ry(-1.5833407277753935) q[0];
ry(-3.044113919936129) q[3];
cx q[0],q[3];
ry(2.83243177188797) q[0];
ry(0.09233237153994711) q[3];
cx q[0],q[3];
ry(2.702628381467471) q[1];
ry(-1.3749635731711125) q[2];
cx q[1],q[2];
ry(2.3537570707530406) q[1];
ry(0.18686762147246813) q[2];
cx q[1],q[2];
ry(0.2501793471073262) q[2];
ry(1.880668564866014) q[5];
cx q[2],q[5];
ry(2.8249302076188165) q[2];
ry(3.068277478037687) q[5];
cx q[2],q[5];
ry(1.6005017728429465) q[3];
ry(0.5667169512006968) q[4];
cx q[3],q[4];
ry(0.07272892179915036) q[3];
ry(1.5670919296939578) q[4];
cx q[3],q[4];
ry(-2.1528858607770855) q[4];
ry(-2.5006124184601104) q[7];
cx q[4],q[7];
ry(-0.032337253288380885) q[4];
ry(-0.29672537100276486) q[7];
cx q[4],q[7];
ry(0.4118911578784787) q[5];
ry(-1.5346236690026331) q[6];
cx q[5],q[6];
ry(0.07591656429222217) q[5];
ry(-2.7889872457978933) q[6];
cx q[5],q[6];
ry(2.98373013153711) q[6];
ry(2.170860772515006) q[9];
cx q[6],q[9];
ry(-0.01281766723631872) q[6];
ry(-0.028710751327413853) q[9];
cx q[6],q[9];
ry(1.3870665656505596) q[7];
ry(-1.700115732139575) q[8];
cx q[7],q[8];
ry(-0.03130635847100738) q[7];
ry(3.1414712328281533) q[8];
cx q[7],q[8];
ry(1.7094490156224542) q[8];
ry(2.1180450863015494) q[11];
cx q[8],q[11];
ry(0.04868702015428881) q[8];
ry(-0.1179985997616324) q[11];
cx q[8],q[11];
ry(1.1442284895076948) q[9];
ry(2.0137719869945734) q[10];
cx q[9],q[10];
ry(0.059650433158954366) q[9];
ry(-0.22248216046419333) q[10];
cx q[9],q[10];
ry(0.39015103129653816) q[10];
ry(-1.8175669818291942) q[13];
cx q[10],q[13];
ry(-3.044750907727322) q[10];
ry(-0.010543497315725365) q[13];
cx q[10],q[13];
ry(2.645433631088014) q[11];
ry(1.7716236265345646) q[12];
cx q[11],q[12];
ry(0.00926883454204752) q[11];
ry(-0.007631677947622428) q[12];
cx q[11],q[12];
ry(-1.1973156931382514) q[12];
ry(-2.3649613981594686) q[15];
cx q[12],q[15];
ry(-0.18180275486378805) q[12];
ry(1.6717324125341193) q[15];
cx q[12],q[15];
ry(2.6542698829226703) q[13];
ry(-2.5128332141046354) q[14];
cx q[13],q[14];
ry(2.9382563542027977) q[13];
ry(2.477668364577484) q[14];
cx q[13],q[14];
ry(0.06499113702421822) q[0];
ry(0.8135629608145112) q[1];
cx q[0],q[1];
ry(1.483172418355415) q[0];
ry(1.7919152696535274) q[1];
cx q[0],q[1];
ry(-2.032398279611363) q[2];
ry(-0.2507876974465759) q[3];
cx q[2],q[3];
ry(0.978458442883051) q[2];
ry(-1.741743966236836) q[3];
cx q[2],q[3];
ry(-1.5106198376506286) q[4];
ry(0.12804338473886467) q[5];
cx q[4],q[5];
ry(-1.6256828180365934) q[4];
ry(-1.6137466840694232) q[5];
cx q[4],q[5];
ry(-1.7298303301230877) q[6];
ry(0.2868760978803717) q[7];
cx q[6],q[7];
ry(0.02472625360302195) q[6];
ry(0.05254455312696821) q[7];
cx q[6],q[7];
ry(-2.7908527019169336) q[8];
ry(2.1552271117810022) q[9];
cx q[8],q[9];
ry(3.0880555498053375) q[8];
ry(0.06960373215063421) q[9];
cx q[8],q[9];
ry(-1.900531095388169) q[10];
ry(-0.2702135886791763) q[11];
cx q[10],q[11];
ry(2.891214314464935) q[10];
ry(-0.13409346816814272) q[11];
cx q[10],q[11];
ry(-2.914278939666303) q[12];
ry(-1.1169513635787862) q[13];
cx q[12],q[13];
ry(-2.253996984254364) q[12];
ry(-0.10446845497900836) q[13];
cx q[12],q[13];
ry(-2.9129285673715297) q[14];
ry(2.3451279350656433) q[15];
cx q[14],q[15];
ry(-0.323097679732716) q[14];
ry(-2.8454743623725998) q[15];
cx q[14],q[15];
ry(-0.55023484563502) q[0];
ry(-2.294795048982915) q[2];
cx q[0],q[2];
ry(1.347066413627358) q[0];
ry(1.0507683446073959) q[2];
cx q[0],q[2];
ry(2.1009674127238815) q[2];
ry(0.06642727516894174) q[4];
cx q[2],q[4];
ry(-2.7931993578814573) q[2];
ry(3.0507602197866435) q[4];
cx q[2],q[4];
ry(-1.5789179884709417) q[4];
ry(-1.308544848735413) q[6];
cx q[4],q[6];
ry(-1.3143608356993477) q[4];
ry(0.2990804996260865) q[6];
cx q[4],q[6];
ry(-3.0606446491106594) q[6];
ry(2.7756169978638736) q[8];
cx q[6],q[8];
ry(-2.7642861864993247) q[6];
ry(-2.7825617742112065) q[8];
cx q[6],q[8];
ry(-1.5803679040531877) q[8];
ry(0.05010356700833271) q[10];
cx q[8],q[10];
ry(1.5634527773792348) q[8];
ry(1.5703110194388836) q[10];
cx q[8],q[10];
ry(-2.953615310999855) q[10];
ry(-0.7190043023564963) q[12];
cx q[10],q[12];
ry(0.39038089536609577) q[10];
ry(-0.37583038952044223) q[12];
cx q[10],q[12];
ry(1.5596984227398643) q[12];
ry(0.6442682504869497) q[14];
cx q[12],q[14];
ry(1.5779967645870536) q[12];
ry(1.5618118447638976) q[14];
cx q[12],q[14];
ry(-0.44661411974655596) q[1];
ry(0.05519967642187141) q[3];
cx q[1],q[3];
ry(0.8226849318194145) q[1];
ry(-1.7925707246464024) q[3];
cx q[1],q[3];
ry(-1.8476402996082795) q[3];
ry(0.09479824567709283) q[5];
cx q[3],q[5];
ry(2.5188674185775306) q[3];
ry(-1.5475438028717632) q[5];
cx q[3],q[5];
ry(0.8499973065694867) q[5];
ry(-2.9675751648730166) q[7];
cx q[5],q[7];
ry(-1.605242225645397) q[5];
ry(3.10062996015535) q[7];
cx q[5],q[7];
ry(2.666175123779153) q[7];
ry(2.3380104877121717) q[9];
cx q[7],q[9];
ry(2.6707951636860288) q[7];
ry(-2.743618516023101) q[9];
cx q[7],q[9];
ry(-1.5808398242818367) q[9];
ry(-1.759238082805657) q[11];
cx q[9],q[11];
ry(1.5702331324495375) q[9];
ry(1.5718555826792082) q[11];
cx q[9],q[11];
ry(-0.7653161357893417) q[11];
ry(0.5773794755977588) q[13];
cx q[11],q[13];
ry(0.5191333763875197) q[11];
ry(-2.693691513881068) q[13];
cx q[11],q[13];
ry(-1.5827238034496747) q[13];
ry(1.7945798813462517) q[15];
cx q[13],q[15];
ry(1.5896604595771977) q[13];
ry(1.5582076531173383) q[15];
cx q[13],q[15];
ry(-1.8940834647418008) q[0];
ry(-2.099133695727163) q[3];
cx q[0],q[3];
ry(-3.099215782519536) q[0];
ry(3.137384870767975) q[3];
cx q[0],q[3];
ry(-0.5375750631314427) q[1];
ry(-1.1346262056797913) q[2];
cx q[1],q[2];
ry(3.0100115542927814) q[1];
ry(-2.852648925346121) q[2];
cx q[1],q[2];
ry(-1.625120946454815) q[2];
ry(1.5701268530279833) q[5];
cx q[2],q[5];
ry(-3.128708332823024) q[2];
ry(-0.10505186781913964) q[5];
cx q[2],q[5];
ry(-2.019750373955886) q[3];
ry(2.9195015693928212) q[4];
cx q[3],q[4];
ry(-0.049251333978104064) q[3];
ry(-3.125381767984536) q[4];
cx q[3],q[4];
ry(-2.561076259954518) q[4];
ry(-1.6420426191018735) q[7];
cx q[4],q[7];
ry(-3.1354979657318816) q[4];
ry(0.012387256061725886) q[7];
cx q[4],q[7];
ry(1.9204398438802235) q[5];
ry(2.4582005587799065) q[6];
cx q[5],q[6];
ry(-0.008235768751071681) q[5];
ry(-3.138941800554185) q[6];
cx q[5],q[6];
ry(2.253832817053166) q[6];
ry(-0.43115933840020154) q[9];
cx q[6],q[9];
ry(-0.004031534274198572) q[6];
ry(-3.1331407694194797) q[9];
cx q[6],q[9];
ry(2.358722423429467) q[7];
ry(-1.505229105665335) q[8];
cx q[7],q[8];
ry(0.0002991371064613446) q[7];
ry(3.139979539883819) q[8];
cx q[7],q[8];
ry(1.7281019367088892) q[8];
ry(2.967866824431767) q[11];
cx q[8],q[11];
ry(-0.0011584325396454393) q[8];
ry(-0.005031620398226799) q[11];
cx q[8],q[11];
ry(-1.757247556067158) q[9];
ry(-2.673646889774702) q[10];
cx q[9],q[10];
ry(-3.136013606226102) q[9];
ry(3.1402988441385156) q[10];
cx q[9],q[10];
ry(0.240007076942315) q[10];
ry(-2.444944677460553) q[13];
cx q[10],q[13];
ry(-0.014256272486258938) q[10];
ry(3.1362874693406284) q[13];
cx q[10],q[13];
ry(1.5161113470290823) q[11];
ry(0.3609284276150801) q[12];
cx q[11],q[12];
ry(0.004308645633870372) q[11];
ry(0.0024365294580706545) q[12];
cx q[11],q[12];
ry(1.6544944100115513) q[12];
ry(-1.002448998034514) q[15];
cx q[12],q[15];
ry(-3.139466564918811) q[12];
ry(3.128606950596208) q[15];
cx q[12],q[15];
ry(-2.0682384088928885) q[13];
ry(2.5636532933458756) q[14];
cx q[13],q[14];
ry(0.009483389834850975) q[13];
ry(-3.124810111798316) q[14];
cx q[13],q[14];
ry(-2.1220617107871504) q[0];
ry(-1.643754669653509) q[1];
ry(1.1210945368437046) q[2];
ry(1.431320551138925) q[3];
ry(-0.34480141056531277) q[4];
ry(2.376949290242095) q[5];
ry(3.06598226491408) q[6];
ry(-1.1067205315396578) q[7];
ry(1.3410610107588363) q[8];
ry(-0.8690930039004696) q[9];
ry(2.757845141441321) q[10];
ry(0.36786164867308574) q[11];
ry(1.8852484553342368) q[12];
ry(0.4936773928521498) q[13];
ry(-1.4026834089074558) q[14];
ry(1.9422112884140397) q[15];