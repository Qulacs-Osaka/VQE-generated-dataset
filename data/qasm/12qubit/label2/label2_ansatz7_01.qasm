OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.0871464703796123) q[0];
ry(0.21946140131548433) q[1];
cx q[0],q[1];
ry(-2.489926827248712) q[0];
ry(3.141203079732648) q[1];
cx q[0],q[1];
ry(1.0477831319766537) q[0];
ry(1.2676970369706912) q[2];
cx q[0],q[2];
ry(-2.353775387469961) q[0];
ry(0.6569971284831757) q[2];
cx q[0],q[2];
ry(1.3279523505684474) q[0];
ry(1.659770518478899) q[3];
cx q[0],q[3];
ry(1.4391584203005985) q[0];
ry(1.098938918443694) q[3];
cx q[0],q[3];
ry(0.9087786288623276) q[0];
ry(-1.7856539973572867) q[4];
cx q[0],q[4];
ry(2.2833630827722073) q[0];
ry(0.685494433175859) q[4];
cx q[0],q[4];
ry(-1.1320281023727123) q[0];
ry(-1.341346133938889) q[5];
cx q[0],q[5];
ry(1.4802138125274673) q[0];
ry(2.153393063747717) q[5];
cx q[0],q[5];
ry(2.1004302239058035) q[0];
ry(-0.32105559598870226) q[6];
cx q[0],q[6];
ry(0.6142687101872806) q[0];
ry(-0.4100676695156533) q[6];
cx q[0],q[6];
ry(-3.0089679742535416) q[0];
ry(-2.4520036004774592) q[7];
cx q[0],q[7];
ry(2.674331047910904) q[0];
ry(2.629734592037971) q[7];
cx q[0],q[7];
ry(-1.9750207598334015) q[0];
ry(1.2104597320121433) q[8];
cx q[0],q[8];
ry(1.3017619989314984) q[0];
ry(0.764145371205931) q[8];
cx q[0],q[8];
ry(-2.1082238202864074) q[0];
ry(2.663301405751982) q[9];
cx q[0],q[9];
ry(0.19340971257378392) q[0];
ry(0.6568926678636249) q[9];
cx q[0],q[9];
ry(-3.0209457357256437) q[0];
ry(-0.8892333884730865) q[10];
cx q[0],q[10];
ry(-2.7610134482224975) q[0];
ry(-1.3231597320194743) q[10];
cx q[0],q[10];
ry(-2.373129587321834) q[0];
ry(0.7790433814176106) q[11];
cx q[0],q[11];
ry(0.7955024209066814) q[0];
ry(1.6142049373615714) q[11];
cx q[0],q[11];
ry(2.4295252955646185) q[1];
ry(2.1626205788791806) q[2];
cx q[1],q[2];
ry(3.0605534821647504) q[1];
ry(-1.9745472085492883) q[2];
cx q[1],q[2];
ry(2.6468019807055887) q[1];
ry(-3.1406364242833775) q[3];
cx q[1],q[3];
ry(1.1415075082065975) q[1];
ry(-0.7070127877756978) q[3];
cx q[1],q[3];
ry(1.9391848800201954) q[1];
ry(1.0082559472888066) q[4];
cx q[1],q[4];
ry(2.1371070356814523) q[1];
ry(-0.802721925025347) q[4];
cx q[1],q[4];
ry(1.6413902807570975) q[1];
ry(-1.2932577619458745) q[5];
cx q[1],q[5];
ry(0.7186360994296991) q[1];
ry(-0.3510129096282151) q[5];
cx q[1],q[5];
ry(1.3242827145280875) q[1];
ry(-2.772653030324682) q[6];
cx q[1],q[6];
ry(0.6508472077895107) q[1];
ry(2.3308992311836856) q[6];
cx q[1],q[6];
ry(-0.09246831932096011) q[1];
ry(1.1965331645335573) q[7];
cx q[1],q[7];
ry(0.6773625899400015) q[1];
ry(0.32302364881170575) q[7];
cx q[1],q[7];
ry(-2.432564886642584) q[1];
ry(1.769547740031149) q[8];
cx q[1],q[8];
ry(-1.093455008029423) q[1];
ry(0.6060375455433383) q[8];
cx q[1],q[8];
ry(-1.2829132065107665) q[1];
ry(-0.6170855698021863) q[9];
cx q[1],q[9];
ry(-0.37694011099708097) q[1];
ry(2.6014296468469005) q[9];
cx q[1],q[9];
ry(-0.20891161888867593) q[1];
ry(-0.5811163298165213) q[10];
cx q[1],q[10];
ry(-2.0426263547562424) q[1];
ry(-2.3294273220033297) q[10];
cx q[1],q[10];
ry(1.1265155575762913) q[1];
ry(-1.438167296933477) q[11];
cx q[1],q[11];
ry(0.49788828522137335) q[1];
ry(1.9622432128597014) q[11];
cx q[1],q[11];
ry(-0.7202950222107488) q[2];
ry(2.732626232232685) q[3];
cx q[2],q[3];
ry(-0.31306073369811516) q[2];
ry(-1.4924130765711778) q[3];
cx q[2],q[3];
ry(1.911789086905386) q[2];
ry(-0.25357074795736784) q[4];
cx q[2],q[4];
ry(1.3577728356698753) q[2];
ry(0.9543123038831766) q[4];
cx q[2],q[4];
ry(0.9889934746699041) q[2];
ry(-1.8996539794141487) q[5];
cx q[2],q[5];
ry(2.803042709643999) q[2];
ry(-2.3940062121566217) q[5];
cx q[2],q[5];
ry(-2.9837784153025244) q[2];
ry(-0.29198179818440734) q[6];
cx q[2],q[6];
ry(1.3735143676725927) q[2];
ry(-0.5109639421968808) q[6];
cx q[2],q[6];
ry(-1.860087771420461) q[2];
ry(-0.9079375992143812) q[7];
cx q[2],q[7];
ry(-2.6693025438724884) q[2];
ry(1.8659479749061727) q[7];
cx q[2],q[7];
ry(2.826046385851864) q[2];
ry(-1.056740842349318) q[8];
cx q[2],q[8];
ry(-2.803872167817275) q[2];
ry(-2.954758154196358) q[8];
cx q[2],q[8];
ry(-1.602315901580508) q[2];
ry(-0.6428632227151283) q[9];
cx q[2],q[9];
ry(-1.264118811773899) q[2];
ry(-3.0997278613557393) q[9];
cx q[2],q[9];
ry(-1.603830637792675) q[2];
ry(1.2065064703680146) q[10];
cx q[2],q[10];
ry(3.0381714167044165) q[2];
ry(-1.0338685797578937) q[10];
cx q[2],q[10];
ry(2.4953770920031975) q[2];
ry(2.4261038277989586) q[11];
cx q[2],q[11];
ry(-0.9717184231353874) q[2];
ry(1.2002004657810312) q[11];
cx q[2],q[11];
ry(2.8800651091460696) q[3];
ry(1.7856666751818973) q[4];
cx q[3],q[4];
ry(-2.813438313371099) q[3];
ry(1.124208379970094) q[4];
cx q[3],q[4];
ry(-0.2743303951348747) q[3];
ry(0.9756337087687316) q[5];
cx q[3],q[5];
ry(-1.3453745792238865) q[3];
ry(2.4661178704805637) q[5];
cx q[3],q[5];
ry(3.0162028207189095) q[3];
ry(3.077449061913245) q[6];
cx q[3],q[6];
ry(1.976183451963264) q[3];
ry(-2.1420085951022685) q[6];
cx q[3],q[6];
ry(1.5933606985275264) q[3];
ry(-2.897925140053195) q[7];
cx q[3],q[7];
ry(-2.822746495156294) q[3];
ry(-2.377653742993725) q[7];
cx q[3],q[7];
ry(0.33977903011571436) q[3];
ry(-0.8966528940127401) q[8];
cx q[3],q[8];
ry(-0.9220398691274705) q[3];
ry(-1.3640336763168701) q[8];
cx q[3],q[8];
ry(-1.1027475171471564) q[3];
ry(2.506924388028656) q[9];
cx q[3],q[9];
ry(-2.0095043383984077) q[3];
ry(0.5180698041674374) q[9];
cx q[3],q[9];
ry(0.8161045918204664) q[3];
ry(-1.3668456270113225) q[10];
cx q[3],q[10];
ry(2.0067824444893274) q[3];
ry(-0.5550300832432793) q[10];
cx q[3],q[10];
ry(0.7621469904582202) q[3];
ry(-3.0262958141756324) q[11];
cx q[3],q[11];
ry(-1.5233850432638156) q[3];
ry(1.99217844201854) q[11];
cx q[3],q[11];
ry(1.3432047718178284) q[4];
ry(3.031546773119438) q[5];
cx q[4],q[5];
ry(0.4567773315095695) q[4];
ry(-2.4765862522626407) q[5];
cx q[4],q[5];
ry(2.332033077822591) q[4];
ry(0.6423898028289798) q[6];
cx q[4],q[6];
ry(-0.600605490177232) q[4];
ry(-2.6404143062667207) q[6];
cx q[4],q[6];
ry(0.01477874479476693) q[4];
ry(-0.7482880760071273) q[7];
cx q[4],q[7];
ry(-1.1073165433191234) q[4];
ry(-2.3757780721975728) q[7];
cx q[4],q[7];
ry(0.5160176317321756) q[4];
ry(0.9288709957974373) q[8];
cx q[4],q[8];
ry(1.363536924545529) q[4];
ry(-1.8387580177296772) q[8];
cx q[4],q[8];
ry(-2.023878832077415) q[4];
ry(-1.8087615153330379) q[9];
cx q[4],q[9];
ry(-1.789768061393584) q[4];
ry(-2.6021518933229) q[9];
cx q[4],q[9];
ry(-0.34689162534489615) q[4];
ry(-2.821408483256451) q[10];
cx q[4],q[10];
ry(-0.21519323761756054) q[4];
ry(-2.0564568105349244) q[10];
cx q[4],q[10];
ry(-0.1177371541732679) q[4];
ry(1.2658024637374314) q[11];
cx q[4],q[11];
ry(-1.3450868706079644) q[4];
ry(0.25281221153153915) q[11];
cx q[4],q[11];
ry(1.8841067709559558) q[5];
ry(-0.053601740054637366) q[6];
cx q[5],q[6];
ry(-1.5022476503915714) q[5];
ry(2.6508331080374194) q[6];
cx q[5],q[6];
ry(-0.7711747732905397) q[5];
ry(2.8025258224487013) q[7];
cx q[5],q[7];
ry(2.3080769044875007) q[5];
ry(2.9684446549057046) q[7];
cx q[5],q[7];
ry(-2.6971940579423417) q[5];
ry(-1.7978186349691887) q[8];
cx q[5],q[8];
ry(-0.42546972617542794) q[5];
ry(1.1740053900481087) q[8];
cx q[5],q[8];
ry(0.24324163805458898) q[5];
ry(2.847228637625783) q[9];
cx q[5],q[9];
ry(2.32724667670534) q[5];
ry(-2.8598026142045505) q[9];
cx q[5],q[9];
ry(-3.0414835041183412) q[5];
ry(2.2889481822452957) q[10];
cx q[5],q[10];
ry(2.456689131762841) q[5];
ry(-0.19999779087950034) q[10];
cx q[5],q[10];
ry(1.8629974021119322) q[5];
ry(2.4038738065871774) q[11];
cx q[5],q[11];
ry(0.35726987675148525) q[5];
ry(-2.520468147546377) q[11];
cx q[5],q[11];
ry(0.6182512037268723) q[6];
ry(-2.7700211054803243) q[7];
cx q[6],q[7];
ry(-1.9705042617316195) q[6];
ry(-0.5664578388698949) q[7];
cx q[6],q[7];
ry(-1.5426165379738714) q[6];
ry(2.7046062391655763) q[8];
cx q[6],q[8];
ry(2.3668222712030236) q[6];
ry(-2.282844393278321) q[8];
cx q[6],q[8];
ry(-2.0088169819917567) q[6];
ry(2.1718390516155117) q[9];
cx q[6],q[9];
ry(-2.7758923282586854) q[6];
ry(1.886374399855348) q[9];
cx q[6],q[9];
ry(-0.11641446591094251) q[6];
ry(1.544452651134172) q[10];
cx q[6],q[10];
ry(1.6348875957908495) q[6];
ry(-2.6667036137428535) q[10];
cx q[6],q[10];
ry(-3.06742766360451) q[6];
ry(-1.9440450276864694) q[11];
cx q[6],q[11];
ry(-2.326079178074569) q[6];
ry(1.3754092789190855) q[11];
cx q[6],q[11];
ry(-1.2114344522115577) q[7];
ry(-0.1153616546540368) q[8];
cx q[7],q[8];
ry(-1.732358048265409) q[7];
ry(0.2694017072302) q[8];
cx q[7],q[8];
ry(0.620608694146366) q[7];
ry(0.22921864030796793) q[9];
cx q[7],q[9];
ry(2.4790377889155337) q[7];
ry(-2.9244048524897246) q[9];
cx q[7],q[9];
ry(-0.38743912151114923) q[7];
ry(-2.469575731701787) q[10];
cx q[7],q[10];
ry(2.504486293971335) q[7];
ry(3.0036798512944904) q[10];
cx q[7],q[10];
ry(-0.4199933089262151) q[7];
ry(-1.1390912073673134) q[11];
cx q[7],q[11];
ry(1.6416598381297751) q[7];
ry(2.5031939201229676) q[11];
cx q[7],q[11];
ry(0.3573661901898016) q[8];
ry(-1.7906798108212272) q[9];
cx q[8],q[9];
ry(-2.0381176832844323) q[8];
ry(-1.4590500230706644) q[9];
cx q[8],q[9];
ry(-1.4386224147128317) q[8];
ry(-0.6322370548008829) q[10];
cx q[8],q[10];
ry(-2.3091972250156685) q[8];
ry(1.791959128708176) q[10];
cx q[8],q[10];
ry(1.9288166458892908) q[8];
ry(2.4636172048385667) q[11];
cx q[8],q[11];
ry(-2.1951888815522542) q[8];
ry(2.8035690517487364) q[11];
cx q[8],q[11];
ry(2.415163007681562) q[9];
ry(1.3897166067315938) q[10];
cx q[9],q[10];
ry(-2.7847658382501654) q[9];
ry(-2.3211735738042627) q[10];
cx q[9],q[10];
ry(1.0843174043292532) q[9];
ry(-1.2868110153296595) q[11];
cx q[9],q[11];
ry(-0.31105812525905907) q[9];
ry(1.7905781965882746) q[11];
cx q[9],q[11];
ry(-2.2934971820556513) q[10];
ry(-2.7899176560964034) q[11];
cx q[10],q[11];
ry(-1.1306425227211485) q[10];
ry(-1.5848338982921462) q[11];
cx q[10],q[11];
ry(-0.5073950710567336) q[0];
ry(-1.7247455959824594) q[1];
cx q[0],q[1];
ry(2.843754893837456) q[0];
ry(2.7926928929599617) q[1];
cx q[0],q[1];
ry(-2.628712013675447) q[0];
ry(-0.7526818297189548) q[2];
cx q[0],q[2];
ry(-0.3288581000484978) q[0];
ry(1.5709540310711132) q[2];
cx q[0],q[2];
ry(2.314231123631455) q[0];
ry(-3.0836478611797133) q[3];
cx q[0],q[3];
ry(1.0533653261929625) q[0];
ry(0.2516921269043957) q[3];
cx q[0],q[3];
ry(-2.9722003260656686) q[0];
ry(0.451389200052737) q[4];
cx q[0],q[4];
ry(-2.530799590571233) q[0];
ry(-2.8037189870117434) q[4];
cx q[0],q[4];
ry(-0.3139728603093719) q[0];
ry(3.102957693979169) q[5];
cx q[0],q[5];
ry(0.3501604606918498) q[0];
ry(0.20250730634099318) q[5];
cx q[0],q[5];
ry(2.93458950108065) q[0];
ry(1.103149952681651) q[6];
cx q[0],q[6];
ry(-2.366074472229811) q[0];
ry(-2.511084484325328) q[6];
cx q[0],q[6];
ry(0.42710319036598765) q[0];
ry(-1.7761532955191504) q[7];
cx q[0],q[7];
ry(-0.43203767556836986) q[0];
ry(-3.0628953669910333) q[7];
cx q[0],q[7];
ry(2.7703111434916243) q[0];
ry(-0.39816103065616115) q[8];
cx q[0],q[8];
ry(-0.4150400882369141) q[0];
ry(2.3615060047879894) q[8];
cx q[0],q[8];
ry(0.7942462969195071) q[0];
ry(-2.7022569115635413) q[9];
cx q[0],q[9];
ry(1.8855919259972285) q[0];
ry(-2.608746795753836) q[9];
cx q[0],q[9];
ry(-0.7868980246718769) q[0];
ry(1.4889671234267912) q[10];
cx q[0],q[10];
ry(-0.4089168067649032) q[0];
ry(1.9789370128646422) q[10];
cx q[0],q[10];
ry(0.4681179161532727) q[0];
ry(1.8188611925195382) q[11];
cx q[0],q[11];
ry(2.6185974796104325) q[0];
ry(2.058993164970383) q[11];
cx q[0],q[11];
ry(-2.139682013053066) q[1];
ry(1.5283811794608715) q[2];
cx q[1],q[2];
ry(-0.6520462850161852) q[1];
ry(1.0105142629827368) q[2];
cx q[1],q[2];
ry(2.7867196242987875) q[1];
ry(-0.29741021357768815) q[3];
cx q[1],q[3];
ry(-1.8105024549581845) q[1];
ry(-2.6870039181796193) q[3];
cx q[1],q[3];
ry(-2.6450948903303693) q[1];
ry(0.6015164235788237) q[4];
cx q[1],q[4];
ry(-2.5879734856835084) q[1];
ry(-1.2115350753380145) q[4];
cx q[1],q[4];
ry(1.5350822940175561) q[1];
ry(-2.277522701807637) q[5];
cx q[1],q[5];
ry(1.8869133286690536) q[1];
ry(0.7461737652903375) q[5];
cx q[1],q[5];
ry(0.18627585197473007) q[1];
ry(-2.0849785451456913) q[6];
cx q[1],q[6];
ry(1.1626035597975979) q[1];
ry(-0.7338589289589921) q[6];
cx q[1],q[6];
ry(-2.582609519850374) q[1];
ry(1.7525973099162393) q[7];
cx q[1],q[7];
ry(1.8542696589019185) q[1];
ry(-2.6187841056144143) q[7];
cx q[1],q[7];
ry(-0.6346845929100379) q[1];
ry(-1.1811847005703247) q[8];
cx q[1],q[8];
ry(-0.7342227742095186) q[1];
ry(-2.495323026628569) q[8];
cx q[1],q[8];
ry(-2.2137899742458576) q[1];
ry(-1.7400023193438123) q[9];
cx q[1],q[9];
ry(0.6561624683688798) q[1];
ry(1.5670247212178197) q[9];
cx q[1],q[9];
ry(1.8753191239189781) q[1];
ry(2.9162876911613926) q[10];
cx q[1],q[10];
ry(-2.109259666589498) q[1];
ry(-2.2834722790385467) q[10];
cx q[1],q[10];
ry(-2.237229731580859) q[1];
ry(1.2151326536744569) q[11];
cx q[1],q[11];
ry(0.09711462045717634) q[1];
ry(-1.9485215634591766) q[11];
cx q[1],q[11];
ry(3.087244223809168) q[2];
ry(-2.5256658747150444) q[3];
cx q[2],q[3];
ry(2.3036422584079927) q[2];
ry(-2.172215766627473) q[3];
cx q[2],q[3];
ry(-2.8212182657578664) q[2];
ry(0.6399260836019041) q[4];
cx q[2],q[4];
ry(0.5534056066812848) q[2];
ry(-2.779009105581401) q[4];
cx q[2],q[4];
ry(-2.8835539714165193) q[2];
ry(-0.8856216226560191) q[5];
cx q[2],q[5];
ry(1.9605218698533555) q[2];
ry(1.396492613516376) q[5];
cx q[2],q[5];
ry(2.7116951292234077) q[2];
ry(0.4761893790916707) q[6];
cx q[2],q[6];
ry(-1.897171749740033) q[2];
ry(-2.9746256450764466) q[6];
cx q[2],q[6];
ry(0.28200223902818244) q[2];
ry(-1.9400420193150643) q[7];
cx q[2],q[7];
ry(-0.8762841618577933) q[2];
ry(2.946304282687619) q[7];
cx q[2],q[7];
ry(-0.9069421590124831) q[2];
ry(1.0925293514994687) q[8];
cx q[2],q[8];
ry(0.8624250727439922) q[2];
ry(-0.44456530010406775) q[8];
cx q[2],q[8];
ry(-2.976954794515722) q[2];
ry(-1.6084048936755826) q[9];
cx q[2],q[9];
ry(-0.7093704407526902) q[2];
ry(1.5705337246721411) q[9];
cx q[2],q[9];
ry(-0.03807588359404153) q[2];
ry(-1.5643160751858876) q[10];
cx q[2],q[10];
ry(-2.8053159795311045) q[2];
ry(2.5624408980953337) q[10];
cx q[2],q[10];
ry(1.1725220958765092) q[2];
ry(-2.5307380016387637) q[11];
cx q[2],q[11];
ry(-0.8355697530325817) q[2];
ry(2.4381528377011374) q[11];
cx q[2],q[11];
ry(-0.2750305797294287) q[3];
ry(0.7667003896399942) q[4];
cx q[3],q[4];
ry(-1.6818701896295518) q[3];
ry(0.4319303454963057) q[4];
cx q[3],q[4];
ry(-0.8986417724631659) q[3];
ry(-2.4977626434008857) q[5];
cx q[3],q[5];
ry(-1.5625615407278313) q[3];
ry(-2.3699907767352437) q[5];
cx q[3],q[5];
ry(-0.7651535754842289) q[3];
ry(1.2545036337015363) q[6];
cx q[3],q[6];
ry(2.8202796859913115) q[3];
ry(1.6068542794534981) q[6];
cx q[3],q[6];
ry(-0.8341548031734893) q[3];
ry(0.7679699267099082) q[7];
cx q[3],q[7];
ry(0.4042614530990568) q[3];
ry(1.3105315398909945) q[7];
cx q[3],q[7];
ry(-2.7375287940514212) q[3];
ry(-1.556505458578049) q[8];
cx q[3],q[8];
ry(1.5658650351798453) q[3];
ry(1.5908735691731888) q[8];
cx q[3],q[8];
ry(0.3132429606081706) q[3];
ry(3.0984606499425054) q[9];
cx q[3],q[9];
ry(-0.5113820313665624) q[3];
ry(2.0930149778340885) q[9];
cx q[3],q[9];
ry(1.6078559934527394) q[3];
ry(2.7567607871689903) q[10];
cx q[3],q[10];
ry(0.8238943187731121) q[3];
ry(-1.915567865266386) q[10];
cx q[3],q[10];
ry(0.7860838811480361) q[3];
ry(-2.046638913300556) q[11];
cx q[3],q[11];
ry(2.906820140377853) q[3];
ry(2.708289874648002) q[11];
cx q[3],q[11];
ry(1.9701188220567074) q[4];
ry(2.8784442829906163) q[5];
cx q[4],q[5];
ry(-1.4124020752126674) q[4];
ry(1.765729779965323) q[5];
cx q[4],q[5];
ry(-0.41394665051241275) q[4];
ry(1.1528089595350273) q[6];
cx q[4],q[6];
ry(-1.8406177991479922) q[4];
ry(0.3084426875928799) q[6];
cx q[4],q[6];
ry(2.07141807158108) q[4];
ry(-1.5879745539283818) q[7];
cx q[4],q[7];
ry(0.28103098074032146) q[4];
ry(-2.459266585298682) q[7];
cx q[4],q[7];
ry(-0.4740999415530717) q[4];
ry(-0.9210726680421306) q[8];
cx q[4],q[8];
ry(-0.17594697838526968) q[4];
ry(2.2858101071272676) q[8];
cx q[4],q[8];
ry(-0.3535073998141174) q[4];
ry(2.7495515620054425) q[9];
cx q[4],q[9];
ry(-0.7566698659107157) q[4];
ry(0.5355180540856166) q[9];
cx q[4],q[9];
ry(1.2797727329412458) q[4];
ry(1.131482677041758) q[10];
cx q[4],q[10];
ry(-0.377490794502231) q[4];
ry(1.0012414343689267) q[10];
cx q[4],q[10];
ry(-1.103122990885364) q[4];
ry(0.5889059401305148) q[11];
cx q[4],q[11];
ry(-0.506595512916942) q[4];
ry(2.4746343612146116) q[11];
cx q[4],q[11];
ry(-1.32562040771125) q[5];
ry(0.24004241243842728) q[6];
cx q[5],q[6];
ry(-2.712126876008742) q[5];
ry(1.9141900030250412) q[6];
cx q[5],q[6];
ry(1.7965734556846473) q[5];
ry(2.6093752839534767) q[7];
cx q[5],q[7];
ry(-1.0339373774638938) q[5];
ry(-3.1091952827074727) q[7];
cx q[5],q[7];
ry(2.8349768718183777) q[5];
ry(1.5055557662343064) q[8];
cx q[5],q[8];
ry(2.770977131514291) q[5];
ry(-2.8375712667351993) q[8];
cx q[5],q[8];
ry(-2.9614538888449355) q[5];
ry(-1.4607250917146768) q[9];
cx q[5],q[9];
ry(-2.9404203724157076) q[5];
ry(-2.5563938909113433) q[9];
cx q[5],q[9];
ry(-1.092118925691744) q[5];
ry(-0.8489658345642876) q[10];
cx q[5],q[10];
ry(-1.3487203660870684) q[5];
ry(2.818634955103991) q[10];
cx q[5],q[10];
ry(1.841888901758578) q[5];
ry(1.6735304696589015) q[11];
cx q[5],q[11];
ry(-1.3337040583216841) q[5];
ry(2.656286882953827) q[11];
cx q[5],q[11];
ry(-0.7314293404953242) q[6];
ry(1.0861924483759036) q[7];
cx q[6],q[7];
ry(1.2917149265554866) q[6];
ry(2.066345154502907) q[7];
cx q[6],q[7];
ry(-3.063933066973497) q[6];
ry(2.99781237111515) q[8];
cx q[6],q[8];
ry(-2.958302984107914) q[6];
ry(-3.1143102577879125) q[8];
cx q[6],q[8];
ry(-0.31756846340171385) q[6];
ry(-0.6023338955500588) q[9];
cx q[6],q[9];
ry(2.6770782629271155) q[6];
ry(-0.9077898340629961) q[9];
cx q[6],q[9];
ry(2.5266675421137523) q[6];
ry(-3.0894825875981597) q[10];
cx q[6],q[10];
ry(-0.646611851635457) q[6];
ry(-2.0704639720473192) q[10];
cx q[6],q[10];
ry(1.2554931339716413) q[6];
ry(1.9143528122244646) q[11];
cx q[6],q[11];
ry(-0.9542149986659352) q[6];
ry(2.545794472897618) q[11];
cx q[6],q[11];
ry(-1.9671067077839073) q[7];
ry(0.6004609414599091) q[8];
cx q[7],q[8];
ry(2.987984978429362) q[7];
ry(-2.601875338513182) q[8];
cx q[7],q[8];
ry(2.996306401904236) q[7];
ry(0.6004488252664063) q[9];
cx q[7],q[9];
ry(0.32413193012858876) q[7];
ry(0.7226516343739711) q[9];
cx q[7],q[9];
ry(1.823142103397869) q[7];
ry(-2.2679747264587453) q[10];
cx q[7],q[10];
ry(-1.3742271930299834) q[7];
ry(-2.503052454666663) q[10];
cx q[7],q[10];
ry(-0.7631825979886155) q[7];
ry(2.549295299347744) q[11];
cx q[7],q[11];
ry(-2.6039133516258475) q[7];
ry(1.6258703422847611) q[11];
cx q[7],q[11];
ry(1.8628144820729728) q[8];
ry(-1.1094248151883086) q[9];
cx q[8],q[9];
ry(0.29024762932239595) q[8];
ry(-0.8153939633662417) q[9];
cx q[8],q[9];
ry(1.3754023148759764) q[8];
ry(1.2900653869913468) q[10];
cx q[8],q[10];
ry(-2.92090950660418) q[8];
ry(-2.259209090202657) q[10];
cx q[8],q[10];
ry(-2.4418677371073376) q[8];
ry(1.3098943575857112) q[11];
cx q[8],q[11];
ry(-2.4200270194683347) q[8];
ry(-2.919074285459578) q[11];
cx q[8],q[11];
ry(0.4788618000400371) q[9];
ry(1.4997313788610314) q[10];
cx q[9],q[10];
ry(1.576707113819519) q[9];
ry(-1.9771214052906432) q[10];
cx q[9],q[10];
ry(-3.0970219466248676) q[9];
ry(0.3331429210574436) q[11];
cx q[9],q[11];
ry(1.92777896083564) q[9];
ry(2.2298164855670453) q[11];
cx q[9],q[11];
ry(2.993883589493201) q[10];
ry(-1.7664476025482996) q[11];
cx q[10],q[11];
ry(1.968934498771194) q[10];
ry(2.92544587377601) q[11];
cx q[10],q[11];
ry(-1.9408178744899045) q[0];
ry(-0.70703217392995) q[1];
cx q[0],q[1];
ry(-0.06042683607874011) q[0];
ry(-0.6640957065851465) q[1];
cx q[0],q[1];
ry(-0.3722573570465819) q[0];
ry(-0.49462850243196144) q[2];
cx q[0],q[2];
ry(0.8322840196029345) q[0];
ry(-1.1716010678789612) q[2];
cx q[0],q[2];
ry(-1.3727633584533663) q[0];
ry(-1.88730429896186) q[3];
cx q[0],q[3];
ry(-1.0234054387193052) q[0];
ry(2.3367953183818746) q[3];
cx q[0],q[3];
ry(-2.82256762366214) q[0];
ry(-2.726152416463662) q[4];
cx q[0],q[4];
ry(0.9947517620648432) q[0];
ry(0.957421757070032) q[4];
cx q[0],q[4];
ry(-1.72430424804613) q[0];
ry(1.7423091849781107) q[5];
cx q[0],q[5];
ry(-0.5085729147613139) q[0];
ry(2.4209799831252687) q[5];
cx q[0],q[5];
ry(1.7013528810603367) q[0];
ry(2.3109788039886277) q[6];
cx q[0],q[6];
ry(-1.0623779204681145) q[0];
ry(-2.311153266492011) q[6];
cx q[0],q[6];
ry(1.7936503896376548) q[0];
ry(1.0524814835448906) q[7];
cx q[0],q[7];
ry(-2.837109813323596) q[0];
ry(0.6149801506259668) q[7];
cx q[0],q[7];
ry(0.2603329511814115) q[0];
ry(-0.3534703495037146) q[8];
cx q[0],q[8];
ry(1.3916435739697333) q[0];
ry(0.29023190467106375) q[8];
cx q[0],q[8];
ry(2.1777208791590166) q[0];
ry(2.1326687702939013) q[9];
cx q[0],q[9];
ry(-0.9250226491504543) q[0];
ry(2.5111619794275186) q[9];
cx q[0],q[9];
ry(-0.41199272595370373) q[0];
ry(-2.3770928373247946) q[10];
cx q[0],q[10];
ry(2.507002874575662) q[0];
ry(1.2445253256598976) q[10];
cx q[0],q[10];
ry(-0.1590019076694098) q[0];
ry(-0.26362938590901935) q[11];
cx q[0],q[11];
ry(-2.448467837176346) q[0];
ry(-1.1501328122029122) q[11];
cx q[0],q[11];
ry(-0.5460483651996926) q[1];
ry(2.336516686465253) q[2];
cx q[1],q[2];
ry(-1.7740281863444871) q[1];
ry(-0.3488140316040137) q[2];
cx q[1],q[2];
ry(-1.5793033200999047) q[1];
ry(1.4229077896234996) q[3];
cx q[1],q[3];
ry(-2.908935802080144) q[1];
ry(-1.3252208595654233) q[3];
cx q[1],q[3];
ry(-1.4862936200576156) q[1];
ry(-1.4783764834088802) q[4];
cx q[1],q[4];
ry(0.5494273493552564) q[1];
ry(0.52405050854191) q[4];
cx q[1],q[4];
ry(3.1263323144865627) q[1];
ry(0.4562083562087978) q[5];
cx q[1],q[5];
ry(-2.9601457869430066) q[1];
ry(-2.7210132270815115) q[5];
cx q[1],q[5];
ry(-2.5702131432956614) q[1];
ry(-0.8064672061161392) q[6];
cx q[1],q[6];
ry(-2.515126423867264) q[1];
ry(-1.5897645784838046) q[6];
cx q[1],q[6];
ry(2.7538089995013917) q[1];
ry(2.721106929483331) q[7];
cx q[1],q[7];
ry(-2.989501732103715) q[1];
ry(-1.0130622441312787) q[7];
cx q[1],q[7];
ry(1.9542335114213172) q[1];
ry(-1.9873808544833003) q[8];
cx q[1],q[8];
ry(0.9557169368748788) q[1];
ry(1.3711563414089132) q[8];
cx q[1],q[8];
ry(0.5525666909325699) q[1];
ry(-1.6997282664208702) q[9];
cx q[1],q[9];
ry(0.8137227633392753) q[1];
ry(-1.8039781373810877) q[9];
cx q[1],q[9];
ry(-1.6683429860978138) q[1];
ry(1.059186156205521) q[10];
cx q[1],q[10];
ry(-2.376970740716114) q[1];
ry(0.4395486585544867) q[10];
cx q[1],q[10];
ry(2.8211790980710507) q[1];
ry(-0.4404416781499947) q[11];
cx q[1],q[11];
ry(2.210202634215045) q[1];
ry(1.691572312841641) q[11];
cx q[1],q[11];
ry(-2.5376075874393664) q[2];
ry(0.21219722005853558) q[3];
cx q[2],q[3];
ry(-2.340168684981293) q[2];
ry(-2.957281053313123) q[3];
cx q[2],q[3];
ry(-1.8308446952559945) q[2];
ry(-0.5554232424337624) q[4];
cx q[2],q[4];
ry(1.370515506876983) q[2];
ry(2.5301400588709506) q[4];
cx q[2],q[4];
ry(-2.6542106864385033) q[2];
ry(-3.1141437686479607) q[5];
cx q[2],q[5];
ry(1.7527388276054008) q[2];
ry(-2.5695142377254485) q[5];
cx q[2],q[5];
ry(1.4774612877997102) q[2];
ry(-2.2979341677066882) q[6];
cx q[2],q[6];
ry(2.109233286427174) q[2];
ry(1.6143616428474665) q[6];
cx q[2],q[6];
ry(-0.5992606664955273) q[2];
ry(-0.8035787264036509) q[7];
cx q[2],q[7];
ry(2.38460493223141) q[2];
ry(3.003487683514441) q[7];
cx q[2],q[7];
ry(-2.7781558805378594) q[2];
ry(-2.3432789525768616) q[8];
cx q[2],q[8];
ry(-0.17515021012475174) q[2];
ry(0.7031476202813177) q[8];
cx q[2],q[8];
ry(-1.078822435589176) q[2];
ry(1.9106333232248094) q[9];
cx q[2],q[9];
ry(-1.9935793219158007) q[2];
ry(1.0449065243609494) q[9];
cx q[2],q[9];
ry(-2.304061474806989) q[2];
ry(-1.9990604990691194) q[10];
cx q[2],q[10];
ry(-0.506672560605848) q[2];
ry(-1.7659649665058694) q[10];
cx q[2],q[10];
ry(2.5697345701861165) q[2];
ry(-2.425296864890356) q[11];
cx q[2],q[11];
ry(2.8370660095901443) q[2];
ry(-2.382822902521395) q[11];
cx q[2],q[11];
ry(-0.5812335647779348) q[3];
ry(-1.2251490679272417) q[4];
cx q[3],q[4];
ry(-0.41468030951472057) q[3];
ry(2.0903428611797485) q[4];
cx q[3],q[4];
ry(-2.7926566224483698) q[3];
ry(-0.876815526535438) q[5];
cx q[3],q[5];
ry(0.952347770519971) q[3];
ry(-0.2339156329696408) q[5];
cx q[3],q[5];
ry(1.481026436592229) q[3];
ry(-0.01019817834702419) q[6];
cx q[3],q[6];
ry(-2.261608945600445) q[3];
ry(-0.33055188971945526) q[6];
cx q[3],q[6];
ry(2.529114015727016) q[3];
ry(-2.854612599340227) q[7];
cx q[3],q[7];
ry(-2.42803198899336) q[3];
ry(1.6475175991434892) q[7];
cx q[3],q[7];
ry(-3.0424244053388563) q[3];
ry(-0.5283919770435117) q[8];
cx q[3],q[8];
ry(0.7297615533845327) q[3];
ry(0.712145058676571) q[8];
cx q[3],q[8];
ry(-2.101822622932753) q[3];
ry(-1.6724283127701005) q[9];
cx q[3],q[9];
ry(1.2996153970675897) q[3];
ry(0.9924131587592687) q[9];
cx q[3],q[9];
ry(-0.808523515370986) q[3];
ry(2.05853154269484) q[10];
cx q[3],q[10];
ry(1.984889455039867) q[3];
ry(0.6127441280629782) q[10];
cx q[3],q[10];
ry(0.29874851342913367) q[3];
ry(2.515601146014175) q[11];
cx q[3],q[11];
ry(-2.747341925238498) q[3];
ry(-1.1340363644644573) q[11];
cx q[3],q[11];
ry(-1.8702878044124347) q[4];
ry(2.1947375083258036) q[5];
cx q[4],q[5];
ry(-2.2359310717909913) q[4];
ry(2.284566457483051) q[5];
cx q[4],q[5];
ry(0.5771851522754305) q[4];
ry(-1.960252185913749) q[6];
cx q[4],q[6];
ry(3.054404492652864) q[4];
ry(0.8674935978441695) q[6];
cx q[4],q[6];
ry(-0.2267454991481399) q[4];
ry(0.8430332674157167) q[7];
cx q[4],q[7];
ry(-0.9723575292062563) q[4];
ry(1.3548259627091293) q[7];
cx q[4],q[7];
ry(-2.873330588832053) q[4];
ry(0.5896944151278547) q[8];
cx q[4],q[8];
ry(0.6678018716254457) q[4];
ry(-2.397413281406614) q[8];
cx q[4],q[8];
ry(2.7427830066094345) q[4];
ry(-0.36132755719213794) q[9];
cx q[4],q[9];
ry(-2.441817340358636) q[4];
ry(1.068869915749989) q[9];
cx q[4],q[9];
ry(-0.3390164650977656) q[4];
ry(-0.4756867964329956) q[10];
cx q[4],q[10];
ry(-1.6326028530347838) q[4];
ry(-1.6380643594529536) q[10];
cx q[4],q[10];
ry(-0.22973679491245394) q[4];
ry(-2.657819253097968) q[11];
cx q[4],q[11];
ry(-2.4188523056132674) q[4];
ry(-1.0169827763685388) q[11];
cx q[4],q[11];
ry(-2.6711958553168365) q[5];
ry(0.03983090802778832) q[6];
cx q[5],q[6];
ry(-1.3547828884415614) q[5];
ry(-2.314963345043333) q[6];
cx q[5],q[6];
ry(-0.3630815523098905) q[5];
ry(-2.0275439228849454) q[7];
cx q[5],q[7];
ry(-2.063652151289868) q[5];
ry(0.6932191352812271) q[7];
cx q[5],q[7];
ry(2.195667828509445) q[5];
ry(-2.162232738388118) q[8];
cx q[5],q[8];
ry(-2.293488992106165) q[5];
ry(2.1246018193117004) q[8];
cx q[5],q[8];
ry(-1.587219509920275) q[5];
ry(-2.8782646634835247) q[9];
cx q[5],q[9];
ry(-2.7963073161860725) q[5];
ry(1.696615823208739) q[9];
cx q[5],q[9];
ry(-0.6348442770814824) q[5];
ry(0.8645586485478027) q[10];
cx q[5],q[10];
ry(-2.806055174810037) q[5];
ry(-0.9959179845791883) q[10];
cx q[5],q[10];
ry(-2.049698743688781) q[5];
ry(-1.4456544926230783) q[11];
cx q[5],q[11];
ry(0.8895532392699782) q[5];
ry(-0.2291576601719445) q[11];
cx q[5],q[11];
ry(-0.26629975017458385) q[6];
ry(2.0415452729496035) q[7];
cx q[6],q[7];
ry(1.271548582641433) q[6];
ry(2.8181526881637087) q[7];
cx q[6],q[7];
ry(0.8921746080078194) q[6];
ry(-2.140239682938416) q[8];
cx q[6],q[8];
ry(-2.766635971463802) q[6];
ry(1.0922604427058964) q[8];
cx q[6],q[8];
ry(-0.5960137196366686) q[6];
ry(0.036302220060099706) q[9];
cx q[6],q[9];
ry(0.4957207197309375) q[6];
ry(2.135207216814373) q[9];
cx q[6],q[9];
ry(2.8868277567526452) q[6];
ry(-0.6600986648438564) q[10];
cx q[6],q[10];
ry(-0.407696764544042) q[6];
ry(0.20067812097611204) q[10];
cx q[6],q[10];
ry(-2.2438049476658093) q[6];
ry(1.4364801133784368) q[11];
cx q[6],q[11];
ry(-1.2826111376752696) q[6];
ry(1.1815909116070582) q[11];
cx q[6],q[11];
ry(0.5793616334242193) q[7];
ry(-0.6517105237961163) q[8];
cx q[7],q[8];
ry(3.0032121877054903) q[7];
ry(1.3577857760804783) q[8];
cx q[7],q[8];
ry(3.138969109749901) q[7];
ry(2.1432081089809767) q[9];
cx q[7],q[9];
ry(2.103275798545761) q[7];
ry(0.266973957259582) q[9];
cx q[7],q[9];
ry(-2.501453679067643) q[7];
ry(-0.8470714978559718) q[10];
cx q[7],q[10];
ry(2.476843323252888) q[7];
ry(0.5099030204698201) q[10];
cx q[7],q[10];
ry(-0.2862859917300042) q[7];
ry(-1.8879193836535955) q[11];
cx q[7],q[11];
ry(1.2210103089186184) q[7];
ry(0.8416348821353979) q[11];
cx q[7],q[11];
ry(2.8626325234621817) q[8];
ry(2.834753604904174) q[9];
cx q[8],q[9];
ry(2.5519759249960576) q[8];
ry(0.39877820111787265) q[9];
cx q[8],q[9];
ry(-1.5877361359003406) q[8];
ry(2.2988235392431315) q[10];
cx q[8],q[10];
ry(-0.5920454966862893) q[8];
ry(1.2406576314097892) q[10];
cx q[8],q[10];
ry(0.5677313794923655) q[8];
ry(-2.811482884008865) q[11];
cx q[8],q[11];
ry(-1.3746673351637022) q[8];
ry(-2.590159350750603) q[11];
cx q[8],q[11];
ry(-0.7892722077549134) q[9];
ry(0.07056587519564399) q[10];
cx q[9],q[10];
ry(2.3207156502548902) q[9];
ry(1.599589322597286) q[10];
cx q[9],q[10];
ry(-1.3506261182223487) q[9];
ry(2.848971973866368) q[11];
cx q[9],q[11];
ry(2.183922744632778) q[9];
ry(1.562319997562705) q[11];
cx q[9],q[11];
ry(-1.6245498020627371) q[10];
ry(2.412600582992837) q[11];
cx q[10],q[11];
ry(-0.8595786847706363) q[10];
ry(1.3090655963395175) q[11];
cx q[10],q[11];
ry(1.0436341940524647) q[0];
ry(1.7696327561271765) q[1];
cx q[0],q[1];
ry(-0.9212578431355318) q[0];
ry(-2.390410261667846) q[1];
cx q[0],q[1];
ry(-1.471783630482208) q[0];
ry(2.0461000679350767) q[2];
cx q[0],q[2];
ry(-0.4872056501291429) q[0];
ry(0.6220255962225484) q[2];
cx q[0],q[2];
ry(-0.39739356928380065) q[0];
ry(-0.9673509761305734) q[3];
cx q[0],q[3];
ry(-0.4200051217676092) q[0];
ry(-2.3640936956595477) q[3];
cx q[0],q[3];
ry(2.8095451484599763) q[0];
ry(1.5441421529662387) q[4];
cx q[0],q[4];
ry(0.9740252827977879) q[0];
ry(1.1555769012857704) q[4];
cx q[0],q[4];
ry(2.100371780196493) q[0];
ry(2.357802172912815) q[5];
cx q[0],q[5];
ry(-1.0475472915783808) q[0];
ry(1.2629672854842025) q[5];
cx q[0],q[5];
ry(2.5416877844149877) q[0];
ry(-1.467709973794631) q[6];
cx q[0],q[6];
ry(0.675882113699327) q[0];
ry(-0.7122987545965147) q[6];
cx q[0],q[6];
ry(2.7174985325148304) q[0];
ry(0.5453995479090281) q[7];
cx q[0],q[7];
ry(2.8352680012989353) q[0];
ry(0.2574650242627709) q[7];
cx q[0],q[7];
ry(-2.81478863946203) q[0];
ry(1.9561202591563491) q[8];
cx q[0],q[8];
ry(-2.898532731096252) q[0];
ry(-1.5219324673530699) q[8];
cx q[0],q[8];
ry(-2.9165222854965975) q[0];
ry(0.4658672326636451) q[9];
cx q[0],q[9];
ry(1.3505007843976777) q[0];
ry(0.3716073270167772) q[9];
cx q[0],q[9];
ry(-0.006634396944837373) q[0];
ry(-1.5127300160106534) q[10];
cx q[0],q[10];
ry(2.707978486516648) q[0];
ry(-0.1807269295913473) q[10];
cx q[0],q[10];
ry(-2.705303113867889) q[0];
ry(1.9893961428530125) q[11];
cx q[0],q[11];
ry(-2.3614547283917466) q[0];
ry(-0.5380021377385045) q[11];
cx q[0],q[11];
ry(-2.416255889857396) q[1];
ry(2.2098479299808074) q[2];
cx q[1],q[2];
ry(2.430411894014803) q[1];
ry(2.100575549096106) q[2];
cx q[1],q[2];
ry(-1.6042478036834207) q[1];
ry(1.6379752513623185) q[3];
cx q[1],q[3];
ry(2.8530001174208928) q[1];
ry(-2.2680622470274896) q[3];
cx q[1],q[3];
ry(-2.034356866871038) q[1];
ry(-0.07383605868545458) q[4];
cx q[1],q[4];
ry(1.5714444232903404) q[1];
ry(-2.202435613301271) q[4];
cx q[1],q[4];
ry(2.644791509835746) q[1];
ry(-0.5843871377944646) q[5];
cx q[1],q[5];
ry(2.865371965394356) q[1];
ry(-1.557678320460651) q[5];
cx q[1],q[5];
ry(2.740522267058411) q[1];
ry(3.0898780380051174) q[6];
cx q[1],q[6];
ry(2.5251512572688664) q[1];
ry(2.555214856882142) q[6];
cx q[1],q[6];
ry(1.8012771961888223) q[1];
ry(0.9204186551900114) q[7];
cx q[1],q[7];
ry(2.0060009774669014) q[1];
ry(-3.006899210057183) q[7];
cx q[1],q[7];
ry(2.6184711711900097) q[1];
ry(1.8052644282889647) q[8];
cx q[1],q[8];
ry(2.3101445617532446) q[1];
ry(0.8909423962041472) q[8];
cx q[1],q[8];
ry(3.05873392494734) q[1];
ry(2.5251773857495325) q[9];
cx q[1],q[9];
ry(2.8901352007676415) q[1];
ry(-2.606893397145762) q[9];
cx q[1],q[9];
ry(0.04347650984284628) q[1];
ry(-1.3650527993618444) q[10];
cx q[1],q[10];
ry(-1.3101209576161974) q[1];
ry(2.7225447468306845) q[10];
cx q[1],q[10];
ry(-1.4360376980807477) q[1];
ry(0.07526956479019388) q[11];
cx q[1],q[11];
ry(-0.6298523480336105) q[1];
ry(1.264502740115046) q[11];
cx q[1],q[11];
ry(-1.0522851361134533) q[2];
ry(-0.9535723801553457) q[3];
cx q[2],q[3];
ry(-1.477403824303494) q[2];
ry(1.8398974300290694) q[3];
cx q[2],q[3];
ry(0.776489453368633) q[2];
ry(2.012749134738165) q[4];
cx q[2],q[4];
ry(-2.7564491828464956) q[2];
ry(-2.2514472529126683) q[4];
cx q[2],q[4];
ry(-1.0040607426753028) q[2];
ry(1.2209970189726338) q[5];
cx q[2],q[5];
ry(-1.111222210289919) q[2];
ry(1.1446460757978358) q[5];
cx q[2],q[5];
ry(2.438764929944205) q[2];
ry(3.0782782589756605) q[6];
cx q[2],q[6];
ry(1.08747937390262) q[2];
ry(1.9254196722449244) q[6];
cx q[2],q[6];
ry(1.1961638324249035) q[2];
ry(-1.753184017172612) q[7];
cx q[2],q[7];
ry(2.909165956072102) q[2];
ry(1.242117831593884) q[7];
cx q[2],q[7];
ry(-1.9579694527727767) q[2];
ry(-2.3454356857573706) q[8];
cx q[2],q[8];
ry(-0.7152832091986913) q[2];
ry(0.3716948878089454) q[8];
cx q[2],q[8];
ry(0.39992323843833577) q[2];
ry(0.13921972300980415) q[9];
cx q[2],q[9];
ry(2.0889102493920646) q[2];
ry(2.990733802185467) q[9];
cx q[2],q[9];
ry(0.9807428381622882) q[2];
ry(0.20556672017614314) q[10];
cx q[2],q[10];
ry(-2.8883492769428063) q[2];
ry(-1.4406624959016234) q[10];
cx q[2],q[10];
ry(-1.591898117970977) q[2];
ry(-2.6986235321646133) q[11];
cx q[2],q[11];
ry(1.3911185500392778) q[2];
ry(-1.0052367191598848) q[11];
cx q[2],q[11];
ry(-1.583297193437122) q[3];
ry(-2.78274066416351) q[4];
cx q[3],q[4];
ry(1.1261744996983376) q[3];
ry(-1.7186588863024692) q[4];
cx q[3],q[4];
ry(-2.0897405641690074) q[3];
ry(1.9554053649064436) q[5];
cx q[3],q[5];
ry(1.453728044732541) q[3];
ry(2.1262907422805934) q[5];
cx q[3],q[5];
ry(-0.19012347079219527) q[3];
ry(-0.566980341681238) q[6];
cx q[3],q[6];
ry(2.877060084238174) q[3];
ry(-0.6879561991010945) q[6];
cx q[3],q[6];
ry(0.2678711246725399) q[3];
ry(2.008737303988263) q[7];
cx q[3],q[7];
ry(2.964448199920137) q[3];
ry(-1.1225103529725737) q[7];
cx q[3],q[7];
ry(3.069753831636382) q[3];
ry(1.2917724152920609) q[8];
cx q[3],q[8];
ry(-0.2744131991871138) q[3];
ry(1.53694688760232) q[8];
cx q[3],q[8];
ry(-2.3489703148903627) q[3];
ry(-2.203503988520131) q[9];
cx q[3],q[9];
ry(1.0291637725453953) q[3];
ry(0.22113402371594315) q[9];
cx q[3],q[9];
ry(2.3584558223728913) q[3];
ry(-2.591813952206972) q[10];
cx q[3],q[10];
ry(2.9630385899781433) q[3];
ry(2.6251353063314435) q[10];
cx q[3],q[10];
ry(-1.8627341914552755) q[3];
ry(1.5520628776040999) q[11];
cx q[3],q[11];
ry(2.5292696362460823) q[3];
ry(-1.6724560813383422) q[11];
cx q[3],q[11];
ry(-0.35595463471677385) q[4];
ry(2.8006792444894666) q[5];
cx q[4],q[5];
ry(1.7316751321222987) q[4];
ry(0.5945393784457931) q[5];
cx q[4],q[5];
ry(1.7012712185613168) q[4];
ry(2.134085611290428) q[6];
cx q[4],q[6];
ry(2.5732152050520902) q[4];
ry(1.9159717195300714) q[6];
cx q[4],q[6];
ry(0.34309086255380716) q[4];
ry(1.3800251501016785) q[7];
cx q[4],q[7];
ry(1.0269249575277373) q[4];
ry(1.7244440633901243) q[7];
cx q[4],q[7];
ry(2.0865588215910273) q[4];
ry(0.7077182000083271) q[8];
cx q[4],q[8];
ry(-2.4298388049130395) q[4];
ry(-2.1454685593604204) q[8];
cx q[4],q[8];
ry(-0.18074128054462218) q[4];
ry(-2.308434747791544) q[9];
cx q[4],q[9];
ry(1.3802496047359991) q[4];
ry(2.790393180199729) q[9];
cx q[4],q[9];
ry(-0.27337499554592437) q[4];
ry(-0.32660233517242165) q[10];
cx q[4],q[10];
ry(0.6649172809215926) q[4];
ry(0.44585369730442537) q[10];
cx q[4],q[10];
ry(0.32823409454192376) q[4];
ry(1.1344993026194137) q[11];
cx q[4],q[11];
ry(-2.1936917747748588) q[4];
ry(0.25970041086626106) q[11];
cx q[4],q[11];
ry(2.867320369420073) q[5];
ry(-1.7848091112325362) q[6];
cx q[5],q[6];
ry(1.9711679999932379) q[5];
ry(-0.07676469974791168) q[6];
cx q[5],q[6];
ry(0.08608483507187782) q[5];
ry(-2.0649292729810806) q[7];
cx q[5],q[7];
ry(-0.9260189530967855) q[5];
ry(1.0719461107748103) q[7];
cx q[5],q[7];
ry(-1.1701245114843484) q[5];
ry(-1.3190519960343163) q[8];
cx q[5],q[8];
ry(-2.610835588346755) q[5];
ry(1.9847841978875405) q[8];
cx q[5],q[8];
ry(-1.6848536130772072) q[5];
ry(-1.2169178277444284) q[9];
cx q[5],q[9];
ry(2.6544220037399007) q[5];
ry(2.7343170812106945) q[9];
cx q[5],q[9];
ry(1.9139567166846536) q[5];
ry(2.7007341165933636) q[10];
cx q[5],q[10];
ry(-1.7686793722200909) q[5];
ry(2.821984074452811) q[10];
cx q[5],q[10];
ry(-2.630610419701421) q[5];
ry(3.1128448734445224) q[11];
cx q[5],q[11];
ry(-1.3882641969920542) q[5];
ry(0.323617287630376) q[11];
cx q[5],q[11];
ry(2.1382717793100694) q[6];
ry(2.5119081120553757) q[7];
cx q[6],q[7];
ry(-1.7755382146537944) q[6];
ry(-0.8418994768625517) q[7];
cx q[6],q[7];
ry(-2.7699112098147) q[6];
ry(2.8587587190887116) q[8];
cx q[6],q[8];
ry(0.9074856172173016) q[6];
ry(-2.7689710612378065) q[8];
cx q[6],q[8];
ry(1.4915549423560013) q[6];
ry(-1.9377063894362134) q[9];
cx q[6],q[9];
ry(-2.573360208432569) q[6];
ry(1.0639613028996937) q[9];
cx q[6],q[9];
ry(-0.9253885181508036) q[6];
ry(-0.6313442274101939) q[10];
cx q[6],q[10];
ry(2.1646259682653293) q[6];
ry(-0.8868665063716894) q[10];
cx q[6],q[10];
ry(0.7339109639472091) q[6];
ry(-0.5018532543801362) q[11];
cx q[6],q[11];
ry(-1.0302117290882675) q[6];
ry(0.8322517985384144) q[11];
cx q[6],q[11];
ry(-2.7808138620807976) q[7];
ry(1.0811450290323963) q[8];
cx q[7],q[8];
ry(2.7469505357312363) q[7];
ry(-0.17875570921513156) q[8];
cx q[7],q[8];
ry(1.0139679770407006) q[7];
ry(2.8990833468154737) q[9];
cx q[7],q[9];
ry(2.501396598479503) q[7];
ry(-0.6937097195512463) q[9];
cx q[7],q[9];
ry(1.4481170916742387) q[7];
ry(-2.238437011180264) q[10];
cx q[7],q[10];
ry(1.5896734358919575) q[7];
ry(-2.5085463806836508) q[10];
cx q[7],q[10];
ry(1.265854611713408) q[7];
ry(-0.5085502822357739) q[11];
cx q[7],q[11];
ry(0.8749390898318818) q[7];
ry(1.14471952766778) q[11];
cx q[7],q[11];
ry(-1.57033487265135) q[8];
ry(-1.5169496829217948) q[9];
cx q[8],q[9];
ry(-1.6283885836135414) q[8];
ry(-0.7093560292089641) q[9];
cx q[8],q[9];
ry(-2.117329910558908) q[8];
ry(-0.5074637731805061) q[10];
cx q[8],q[10];
ry(-0.8957945760845325) q[8];
ry(-1.5423042053212557) q[10];
cx q[8],q[10];
ry(-2.884154976301624) q[8];
ry(-2.2559274560573384) q[11];
cx q[8],q[11];
ry(2.7546027662141435) q[8];
ry(2.2380129971201908) q[11];
cx q[8],q[11];
ry(1.8135310498261532) q[9];
ry(2.5937961550848163) q[10];
cx q[9],q[10];
ry(-1.820153337867666) q[9];
ry(-2.358165947286029) q[10];
cx q[9],q[10];
ry(2.9330912342604427) q[9];
ry(0.037632154044213835) q[11];
cx q[9],q[11];
ry(-2.448380167377003) q[9];
ry(-0.3509896685155724) q[11];
cx q[9],q[11];
ry(0.24361729887094424) q[10];
ry(1.8684469581302292) q[11];
cx q[10],q[11];
ry(1.048253007196621) q[10];
ry(-1.2311357773005103) q[11];
cx q[10],q[11];
ry(-2.1914139870076745) q[0];
ry(0.6402970769424143) q[1];
ry(1.3020543078584907) q[2];
ry(2.515208210163249) q[3];
ry(0.7179238825639791) q[4];
ry(0.42316353050776023) q[5];
ry(0.6355668873372994) q[6];
ry(-3.034496409629079) q[7];
ry(-2.4843304670443223) q[8];
ry(-1.448192132217419) q[9];
ry(0.6045270244261598) q[10];
ry(0.460080583034542) q[11];