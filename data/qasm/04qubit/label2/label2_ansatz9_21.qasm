OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.8908110987445818) q[0];
ry(-2.6001083374937384) q[1];
cx q[0],q[1];
ry(0.6477719730414895) q[0];
ry(-2.340573368678054) q[1];
cx q[0],q[1];
ry(-1.514409545807192) q[2];
ry(-1.5002402167206272) q[3];
cx q[2],q[3];
ry(-2.524962053716005) q[2];
ry(2.084074573545004) q[3];
cx q[2],q[3];
ry(1.6935673147065422) q[0];
ry(-1.2930575884099422) q[2];
cx q[0],q[2];
ry(0.6737647969873188) q[0];
ry(-3.0132911379479146) q[2];
cx q[0],q[2];
ry(-0.06402099058352344) q[1];
ry(3.0181274041018873) q[3];
cx q[1],q[3];
ry(2.548321732104738) q[1];
ry(0.37225343383123566) q[3];
cx q[1],q[3];
ry(-1.8389741373659867) q[0];
ry(-0.19535041071677436) q[3];
cx q[0],q[3];
ry(-2.323484922867387) q[0];
ry(0.6515461018266073) q[3];
cx q[0],q[3];
ry(-0.8312261822157954) q[1];
ry(2.9097243122865946) q[2];
cx q[1],q[2];
ry(2.3355686763519605) q[1];
ry(-1.6674928328223224) q[2];
cx q[1],q[2];
ry(-0.641738600347816) q[0];
ry(-1.535612307736014) q[1];
cx q[0],q[1];
ry(3.0666966669247735) q[0];
ry(2.6722380047159677) q[1];
cx q[0],q[1];
ry(1.2945582306454186) q[2];
ry(0.6675656420910137) q[3];
cx q[2],q[3];
ry(0.4422140127476757) q[2];
ry(1.3269479790368326) q[3];
cx q[2],q[3];
ry(-0.7995288444668667) q[0];
ry(1.6741956208592244) q[2];
cx q[0],q[2];
ry(2.4760471560155626) q[0];
ry(-2.5796349862525534) q[2];
cx q[0],q[2];
ry(1.751934906420587) q[1];
ry(2.2049937645839712) q[3];
cx q[1],q[3];
ry(2.0975718373169245) q[1];
ry(2.8655037468266973) q[3];
cx q[1],q[3];
ry(-1.9590402461270982) q[0];
ry(0.007482246098292125) q[3];
cx q[0],q[3];
ry(0.6412925844464848) q[0];
ry(-2.4674442099592517) q[3];
cx q[0],q[3];
ry(0.17336605468155528) q[1];
ry(-2.7414677522969417) q[2];
cx q[1],q[2];
ry(-2.428601437330048) q[1];
ry(-1.9881735244943917) q[2];
cx q[1],q[2];
ry(-0.37991275272513736) q[0];
ry(-1.767441488441456) q[1];
cx q[0],q[1];
ry(-1.162320541760642) q[0];
ry(-2.856813046994258) q[1];
cx q[0],q[1];
ry(0.4172286703433317) q[2];
ry(1.681906808144007) q[3];
cx q[2],q[3];
ry(-1.9507024215683664) q[2];
ry(-0.615869935938025) q[3];
cx q[2],q[3];
ry(0.20413426202835688) q[0];
ry(0.7879772812323793) q[2];
cx q[0],q[2];
ry(0.5474889948584858) q[0];
ry(-1.2921188609147058) q[2];
cx q[0],q[2];
ry(1.2015695813070142) q[1];
ry(2.072008811270055) q[3];
cx q[1],q[3];
ry(-1.3041231704315932) q[1];
ry(-2.8650310830098236) q[3];
cx q[1],q[3];
ry(-0.5622159592118416) q[0];
ry(1.5414988338270341) q[3];
cx q[0],q[3];
ry(-0.7214600726142969) q[0];
ry(0.8069887532756177) q[3];
cx q[0],q[3];
ry(0.687700048768257) q[1];
ry(0.5457917562920419) q[2];
cx q[1],q[2];
ry(2.953307189444647) q[1];
ry(1.993410300585274) q[2];
cx q[1],q[2];
ry(-2.290154784163481) q[0];
ry(1.616530181233318) q[1];
cx q[0],q[1];
ry(1.0348058119810604) q[0];
ry(-0.31805431528138145) q[1];
cx q[0],q[1];
ry(2.322524007474563) q[2];
ry(-0.17078699088494204) q[3];
cx q[2],q[3];
ry(-3.1057593600978537) q[2];
ry(2.3510435666360188) q[3];
cx q[2],q[3];
ry(1.8122956834807218) q[0];
ry(1.5907588246747129) q[2];
cx q[0],q[2];
ry(-2.413920444774203) q[0];
ry(-2.8648272523459455) q[2];
cx q[0],q[2];
ry(-1.8997887504569546) q[1];
ry(0.673420130498042) q[3];
cx q[1],q[3];
ry(-2.8932000387009933) q[1];
ry(-1.7631453229766763) q[3];
cx q[1],q[3];
ry(-0.23958095465024784) q[0];
ry(-0.25409618680934754) q[3];
cx q[0],q[3];
ry(1.6551249455878496) q[0];
ry(-0.4626093576273318) q[3];
cx q[0],q[3];
ry(-1.437988655259618) q[1];
ry(-2.8616008455509836) q[2];
cx q[1],q[2];
ry(-2.9666734062760947) q[1];
ry(-2.2034306897272242) q[2];
cx q[1],q[2];
ry(1.9420368891460615) q[0];
ry(-1.0903345113054181) q[1];
cx q[0],q[1];
ry(-2.7892185460095207) q[0];
ry(-0.5028584677879415) q[1];
cx q[0],q[1];
ry(-1.4030189404189954) q[2];
ry(-0.41475555804296255) q[3];
cx q[2],q[3];
ry(1.513766162760188) q[2];
ry(-2.9258818909728017) q[3];
cx q[2],q[3];
ry(-3.0701317577537357) q[0];
ry(0.6235295283632774) q[2];
cx q[0],q[2];
ry(-1.0048004594917037) q[0];
ry(-0.08988904227867305) q[2];
cx q[0],q[2];
ry(0.5952807952404093) q[1];
ry(1.9104466858632418) q[3];
cx q[1],q[3];
ry(1.8261396635354612) q[1];
ry(0.02803996853690435) q[3];
cx q[1],q[3];
ry(1.7429546192730463) q[0];
ry(-0.9950164633784045) q[3];
cx q[0],q[3];
ry(0.7365551091088633) q[0];
ry(-0.6359424107639127) q[3];
cx q[0],q[3];
ry(0.5432472253021156) q[1];
ry(-1.1385332012373268) q[2];
cx q[1],q[2];
ry(2.2283025379570516) q[1];
ry(2.5786321533265504) q[2];
cx q[1],q[2];
ry(-2.973087584067428) q[0];
ry(-0.9550156299998838) q[1];
cx q[0],q[1];
ry(1.247037872138863) q[0];
ry(1.9467977654186077) q[1];
cx q[0],q[1];
ry(2.622290882091266) q[2];
ry(-2.801596382879419) q[3];
cx q[2],q[3];
ry(2.5858398923493007) q[2];
ry(-2.516482642514442) q[3];
cx q[2],q[3];
ry(-2.1988306020997883) q[0];
ry(-1.617901870049273) q[2];
cx q[0],q[2];
ry(-1.4284673010950089) q[0];
ry(0.5537601037929969) q[2];
cx q[0],q[2];
ry(1.7923424864152426) q[1];
ry(0.478097315708502) q[3];
cx q[1],q[3];
ry(2.8590317611728495) q[1];
ry(-1.3568653276620726) q[3];
cx q[1],q[3];
ry(-0.12962074593554274) q[0];
ry(2.5949134430852068) q[3];
cx q[0],q[3];
ry(-2.268538263684766) q[0];
ry(-0.9123901544090282) q[3];
cx q[0],q[3];
ry(1.3440161706936764) q[1];
ry(0.25927122930878976) q[2];
cx q[1],q[2];
ry(1.1847750026438417) q[1];
ry(-0.38553515711914343) q[2];
cx q[1],q[2];
ry(1.9962598054004048) q[0];
ry(-0.028467026425427585) q[1];
cx q[0],q[1];
ry(-1.4077668518456699) q[0];
ry(-0.9665915413796407) q[1];
cx q[0],q[1];
ry(2.8169143315373795) q[2];
ry(-1.0312272863395382) q[3];
cx q[2],q[3];
ry(-1.2737741896349282) q[2];
ry(-0.2773789391211334) q[3];
cx q[2],q[3];
ry(-1.7511751843673562) q[0];
ry(0.8163482276864903) q[2];
cx q[0],q[2];
ry(-0.6724203152859621) q[0];
ry(0.24997822805250627) q[2];
cx q[0],q[2];
ry(-1.0443085373346523) q[1];
ry(1.4940494665647224) q[3];
cx q[1],q[3];
ry(-2.6724856890635653) q[1];
ry(-2.2294223490097402) q[3];
cx q[1],q[3];
ry(0.35522289441686095) q[0];
ry(-2.06472679716676) q[3];
cx q[0],q[3];
ry(2.9015305949576193) q[0];
ry(2.8782697417931358) q[3];
cx q[0],q[3];
ry(-0.19433602028275487) q[1];
ry(-2.6343400277585047) q[2];
cx q[1],q[2];
ry(0.9555752230766286) q[1];
ry(0.7473971628930931) q[2];
cx q[1],q[2];
ry(1.7004567002107036) q[0];
ry(3.0471490276632944) q[1];
cx q[0],q[1];
ry(0.3772890021879924) q[0];
ry(-2.6962032457182312) q[1];
cx q[0],q[1];
ry(1.431698031363393) q[2];
ry(2.2807975228457447) q[3];
cx q[2],q[3];
ry(0.22922275433120376) q[2];
ry(0.574822954619302) q[3];
cx q[2],q[3];
ry(-1.4987805611512846) q[0];
ry(-2.2068229141683884) q[2];
cx q[0],q[2];
ry(0.8118832656368533) q[0];
ry(0.041150688024738535) q[2];
cx q[0],q[2];
ry(-1.5532408006176224) q[1];
ry(-2.8190093213046463) q[3];
cx q[1],q[3];
ry(2.0567979606247695) q[1];
ry(1.9405926112010894) q[3];
cx q[1],q[3];
ry(-2.3789726394174835) q[0];
ry(-2.559362863308542) q[3];
cx q[0],q[3];
ry(-1.7123358496693752) q[0];
ry(2.799136597231928) q[3];
cx q[0],q[3];
ry(-0.0990443514344852) q[1];
ry(2.1153079087529463) q[2];
cx q[1],q[2];
ry(1.5990000514592042) q[1];
ry(-1.4162125191344714) q[2];
cx q[1],q[2];
ry(2.941986336721817) q[0];
ry(1.6485782027564504) q[1];
cx q[0],q[1];
ry(0.30795119695065765) q[0];
ry(-2.182521122708183) q[1];
cx q[0],q[1];
ry(-0.595794574641225) q[2];
ry(-2.7764548076989652) q[3];
cx q[2],q[3];
ry(-0.8406713158681907) q[2];
ry(-2.8774429718000962) q[3];
cx q[2],q[3];
ry(0.9321747038655808) q[0];
ry(-1.6697189188436488) q[2];
cx q[0],q[2];
ry(1.526972481053938) q[0];
ry(0.14781197783204458) q[2];
cx q[0],q[2];
ry(1.453231590883699) q[1];
ry(0.22122359982644096) q[3];
cx q[1],q[3];
ry(0.3610683067857018) q[1];
ry(1.713175874354844) q[3];
cx q[1],q[3];
ry(0.7491827275475679) q[0];
ry(-1.3461693046077914) q[3];
cx q[0],q[3];
ry(-1.8787143814366152) q[0];
ry(1.5171165998043579) q[3];
cx q[0],q[3];
ry(0.7517741945962585) q[1];
ry(-3.03535978036092) q[2];
cx q[1],q[2];
ry(-0.27266036426141393) q[1];
ry(0.8762492054272363) q[2];
cx q[1],q[2];
ry(-0.1537014501704519) q[0];
ry(0.7960607693259032) q[1];
cx q[0],q[1];
ry(1.4815646382071364) q[0];
ry(-2.8590719115423586) q[1];
cx q[0],q[1];
ry(-0.5066619426775523) q[2];
ry(2.304623360324566) q[3];
cx q[2],q[3];
ry(-1.3323812613348007) q[2];
ry(-2.9485337842011035) q[3];
cx q[2],q[3];
ry(-2.6418805787685082) q[0];
ry(-2.7800615929520607) q[2];
cx q[0],q[2];
ry(-2.7009893243823253) q[0];
ry(1.0720173241553062) q[2];
cx q[0],q[2];
ry(1.634184651509434) q[1];
ry(-0.8381918964199259) q[3];
cx q[1],q[3];
ry(3.0191601259881558) q[1];
ry(-0.9264510711866691) q[3];
cx q[1],q[3];
ry(-2.477811919776504) q[0];
ry(-0.7506880297557936) q[3];
cx q[0],q[3];
ry(-1.672625834178658) q[0];
ry(-0.20314664942685173) q[3];
cx q[0],q[3];
ry(-1.1713507185075276) q[1];
ry(-2.8196008981293965) q[2];
cx q[1],q[2];
ry(-2.667898541735948) q[1];
ry(-1.6999878574318672) q[2];
cx q[1],q[2];
ry(-2.595151654736934) q[0];
ry(1.2104319860225723) q[1];
cx q[0],q[1];
ry(-0.4270542444014667) q[0];
ry(3.1210138450575857) q[1];
cx q[0],q[1];
ry(0.8642831397059475) q[2];
ry(-2.966442956996747) q[3];
cx q[2],q[3];
ry(-0.4478336498016804) q[2];
ry(0.4241910500107484) q[3];
cx q[2],q[3];
ry(0.45356873947863435) q[0];
ry(-0.4126096830889018) q[2];
cx q[0],q[2];
ry(-0.3794054542980381) q[0];
ry(-2.262872938995696) q[2];
cx q[0],q[2];
ry(2.512520393029047) q[1];
ry(2.593774131183074) q[3];
cx q[1],q[3];
ry(2.320193387395159) q[1];
ry(2.0608858473300042) q[3];
cx q[1],q[3];
ry(-2.1808908794868014) q[0];
ry(-0.6045673601820221) q[3];
cx q[0],q[3];
ry(2.683423423256352) q[0];
ry(-2.875695046001197) q[3];
cx q[0],q[3];
ry(-1.974570595718168) q[1];
ry(-1.6859765798922854) q[2];
cx q[1],q[2];
ry(1.0154398396276323) q[1];
ry(-2.798757800001042) q[2];
cx q[1],q[2];
ry(0.15866942328044065) q[0];
ry(-0.9168645562286201) q[1];
cx q[0],q[1];
ry(1.7355507730471726) q[0];
ry(-2.6706993206089225) q[1];
cx q[0],q[1];
ry(2.5723231880307407) q[2];
ry(2.4975725580480312) q[3];
cx q[2],q[3];
ry(-1.4527316532132104) q[2];
ry(1.2881621709268891) q[3];
cx q[2],q[3];
ry(-0.4978511442907321) q[0];
ry(0.2964805148356688) q[2];
cx q[0],q[2];
ry(0.6417163975896968) q[0];
ry(3.082116195863166) q[2];
cx q[0],q[2];
ry(1.5743207202409817) q[1];
ry(-2.943273421931566) q[3];
cx q[1],q[3];
ry(0.7306300811744048) q[1];
ry(0.3727379898518512) q[3];
cx q[1],q[3];
ry(-2.3484562629256764) q[0];
ry(1.6861830234885005) q[3];
cx q[0],q[3];
ry(-2.546748984580803) q[0];
ry(-1.9845527107790037) q[3];
cx q[0],q[3];
ry(0.04275586373894846) q[1];
ry(1.3506148917292915) q[2];
cx q[1],q[2];
ry(-2.947493245480426) q[1];
ry(-2.571823389631399) q[2];
cx q[1],q[2];
ry(0.4774738584290438) q[0];
ry(1.0492729880606992) q[1];
cx q[0],q[1];
ry(1.3871395930411126) q[0];
ry(-1.4658426295227227) q[1];
cx q[0],q[1];
ry(-0.6395517794961787) q[2];
ry(2.8753007402270607) q[3];
cx q[2],q[3];
ry(2.902746650898613) q[2];
ry(1.9084239946664256) q[3];
cx q[2],q[3];
ry(2.1570287289322954) q[0];
ry(-1.067009810148262) q[2];
cx q[0],q[2];
ry(-1.3308741394589807) q[0];
ry(2.971940859468858) q[2];
cx q[0],q[2];
ry(2.8050526202758688) q[1];
ry(-2.6130462053947623) q[3];
cx q[1],q[3];
ry(-0.07953297270754067) q[1];
ry(2.053224830194055) q[3];
cx q[1],q[3];
ry(0.35254861958527517) q[0];
ry(-2.3985280864107636) q[3];
cx q[0],q[3];
ry(2.7767311728434447) q[0];
ry(-0.8435036969365086) q[3];
cx q[0],q[3];
ry(0.0740757673462431) q[1];
ry(2.3408009266514807) q[2];
cx q[1],q[2];
ry(2.7756739322208963) q[1];
ry(-2.904362180492837) q[2];
cx q[1],q[2];
ry(2.6196356436000494) q[0];
ry(0.23717600769440758) q[1];
cx q[0],q[1];
ry(-1.5193070039669356) q[0];
ry(1.4684282201678995) q[1];
cx q[0],q[1];
ry(0.03561254845675066) q[2];
ry(-0.3497027248577931) q[3];
cx q[2],q[3];
ry(2.3252093076427784) q[2];
ry(1.8105599903464304) q[3];
cx q[2],q[3];
ry(1.5496401356283611) q[0];
ry(-0.3289183521395769) q[2];
cx q[0],q[2];
ry(0.30891937061620833) q[0];
ry(0.9540000505699693) q[2];
cx q[0],q[2];
ry(-1.5028160262108443) q[1];
ry(-0.6359427616166321) q[3];
cx q[1],q[3];
ry(-0.7414311110963787) q[1];
ry(-1.0745260987223508) q[3];
cx q[1],q[3];
ry(-0.6971000272075619) q[0];
ry(2.9352630434595812) q[3];
cx q[0],q[3];
ry(-2.8860664664812856) q[0];
ry(-0.3853415768375079) q[3];
cx q[0],q[3];
ry(0.3004999282361143) q[1];
ry(-3.0210683362957016) q[2];
cx q[1],q[2];
ry(0.9956110693579343) q[1];
ry(2.5984443815756926) q[2];
cx q[1],q[2];
ry(-1.139495623279256) q[0];
ry(-0.1654004689171147) q[1];
cx q[0],q[1];
ry(1.729326218408788) q[0];
ry(0.9841576210331007) q[1];
cx q[0],q[1];
ry(1.2372471292349339) q[2];
ry(0.6571660880382066) q[3];
cx q[2],q[3];
ry(-2.0984995344790454) q[2];
ry(2.069124425683876) q[3];
cx q[2],q[3];
ry(-0.6438015561370154) q[0];
ry(2.5693759727528143) q[2];
cx q[0],q[2];
ry(-0.2814521724900488) q[0];
ry(-0.11215298534488483) q[2];
cx q[0],q[2];
ry(-3.0094801491567877) q[1];
ry(-1.8755176665854982) q[3];
cx q[1],q[3];
ry(-1.2898716403806336) q[1];
ry(2.806710434181807) q[3];
cx q[1],q[3];
ry(-0.914853364256488) q[0];
ry(-1.4057431078257734) q[3];
cx q[0],q[3];
ry(-1.5401064375574967) q[0];
ry(1.2868725327086703) q[3];
cx q[0],q[3];
ry(-0.8800637502811552) q[1];
ry(1.1093970169861045) q[2];
cx q[1],q[2];
ry(0.6563267828401393) q[1];
ry(-1.2902005214062289) q[2];
cx q[1],q[2];
ry(-2.9668598585274575) q[0];
ry(2.7756521913873984) q[1];
cx q[0],q[1];
ry(0.7462331255761461) q[0];
ry(-2.691826402023875) q[1];
cx q[0],q[1];
ry(2.9297245612134155) q[2];
ry(-3.130631787987223) q[3];
cx q[2],q[3];
ry(1.193440331431539) q[2];
ry(-2.929767183330464) q[3];
cx q[2],q[3];
ry(0.39791876724093517) q[0];
ry(-2.7280128338295273) q[2];
cx q[0],q[2];
ry(-0.16799686056355864) q[0];
ry(-0.9659266236435018) q[2];
cx q[0],q[2];
ry(1.204978264073997) q[1];
ry(-0.4041237829333264) q[3];
cx q[1],q[3];
ry(2.9562225565675844) q[1];
ry(-1.444989446809151) q[3];
cx q[1],q[3];
ry(2.712272827319684) q[0];
ry(-1.92195783386354) q[3];
cx q[0],q[3];
ry(2.8984857499782253) q[0];
ry(-2.807715109435553) q[3];
cx q[0],q[3];
ry(-1.7327070178459199) q[1];
ry(-2.3752622402528023) q[2];
cx q[1],q[2];
ry(-1.4186902422256544) q[1];
ry(-0.2581232560124924) q[2];
cx q[1],q[2];
ry(-2.861664159039074) q[0];
ry(2.9332139789734573) q[1];
cx q[0],q[1];
ry(3.000997154911108) q[0];
ry(-2.695874366987133) q[1];
cx q[0],q[1];
ry(-2.0816001181593986) q[2];
ry(0.8403098176298779) q[3];
cx q[2],q[3];
ry(-1.4002239533262908) q[2];
ry(1.4833558508911182) q[3];
cx q[2],q[3];
ry(-1.3029290136872536) q[0];
ry(0.5722859187299133) q[2];
cx q[0],q[2];
ry(-0.9803450464974933) q[0];
ry(1.4277183537064886) q[2];
cx q[0],q[2];
ry(1.1123345194300809) q[1];
ry(-1.170610951938528) q[3];
cx q[1],q[3];
ry(-0.09414597607360786) q[1];
ry(1.9003228373176408) q[3];
cx q[1],q[3];
ry(-1.275559911599232) q[0];
ry(0.003629694480743574) q[3];
cx q[0],q[3];
ry(2.2645269379440536) q[0];
ry(1.4295548756182945) q[3];
cx q[0],q[3];
ry(-1.311163206381442) q[1];
ry(-1.9299349897190243) q[2];
cx q[1],q[2];
ry(1.3899295262540559) q[1];
ry(1.9953748120568404) q[2];
cx q[1],q[2];
ry(0.13070283295384807) q[0];
ry(-0.5654483949814759) q[1];
cx q[0],q[1];
ry(2.9785635697086508) q[0];
ry(0.43791138860498696) q[1];
cx q[0],q[1];
ry(-2.374935039464462) q[2];
ry(1.9091111940861263) q[3];
cx q[2],q[3];
ry(2.795426759299029) q[2];
ry(-1.1746293551612048) q[3];
cx q[2],q[3];
ry(-0.7853758768316563) q[0];
ry(1.3492600611517531) q[2];
cx q[0],q[2];
ry(2.6601293368575565) q[0];
ry(-1.2433962109458019) q[2];
cx q[0],q[2];
ry(-2.1708106153166415) q[1];
ry(1.4153547444451675) q[3];
cx q[1],q[3];
ry(-1.9697403132149107) q[1];
ry(1.7983985528880224) q[3];
cx q[1],q[3];
ry(0.3103295651960654) q[0];
ry(-1.9493886258804045) q[3];
cx q[0],q[3];
ry(3.0244900331807045) q[0];
ry(-1.820302755158232) q[3];
cx q[0],q[3];
ry(0.5171195470803697) q[1];
ry(2.684057812971104) q[2];
cx q[1],q[2];
ry(2.58401627351577) q[1];
ry(-1.2586402337758775) q[2];
cx q[1],q[2];
ry(-0.3819526367850008) q[0];
ry(0.6308712335989755) q[1];
cx q[0],q[1];
ry(1.204285293103501) q[0];
ry(-0.14004813499548313) q[1];
cx q[0],q[1];
ry(0.7214852435952553) q[2];
ry(-0.5689939572497719) q[3];
cx q[2],q[3];
ry(-2.775813783636359) q[2];
ry(1.383922696036176) q[3];
cx q[2],q[3];
ry(2.1512420734028526) q[0];
ry(0.6213431617607288) q[2];
cx q[0],q[2];
ry(-1.1355305386339563) q[0];
ry(1.8413934446017273) q[2];
cx q[0],q[2];
ry(-1.3732445170062997) q[1];
ry(0.3474129351053844) q[3];
cx q[1],q[3];
ry(1.686735800248368) q[1];
ry(0.31900077785755043) q[3];
cx q[1],q[3];
ry(0.037049910576989066) q[0];
ry(1.0718387750402885) q[3];
cx q[0],q[3];
ry(-1.028371180849443) q[0];
ry(-0.5879454289952344) q[3];
cx q[0],q[3];
ry(0.7964785427702559) q[1];
ry(-1.7149985586612946) q[2];
cx q[1],q[2];
ry(2.231839732373912) q[1];
ry(2.178232460227517) q[2];
cx q[1],q[2];
ry(-2.175923797086668) q[0];
ry(0.43054639191004684) q[1];
cx q[0],q[1];
ry(2.269335892308745) q[0];
ry(-1.5169927556824951) q[1];
cx q[0],q[1];
ry(-1.9353428768361605) q[2];
ry(-1.6922532314554752) q[3];
cx q[2],q[3];
ry(0.7882275202312128) q[2];
ry(1.7454873287785864) q[3];
cx q[2],q[3];
ry(-0.7913521818554774) q[0];
ry(0.3266271873007102) q[2];
cx q[0],q[2];
ry(-2.8654992633258067) q[0];
ry(-2.2577602312816354) q[2];
cx q[0],q[2];
ry(-0.45216519894379026) q[1];
ry(-0.5115703309111455) q[3];
cx q[1],q[3];
ry(-1.4428186994618608) q[1];
ry(-2.0740425022807645) q[3];
cx q[1],q[3];
ry(2.2592569784587266) q[0];
ry(0.5964824204972228) q[3];
cx q[0],q[3];
ry(-2.9257280563644406) q[0];
ry(-0.004120356616035714) q[3];
cx q[0],q[3];
ry(-1.6417670616880735) q[1];
ry(2.4917895772227667) q[2];
cx q[1],q[2];
ry(-2.350985367236094) q[1];
ry(-0.6332884319809449) q[2];
cx q[1],q[2];
ry(-0.2045507117255337) q[0];
ry(-2.7320284669366) q[1];
cx q[0],q[1];
ry(0.04860543724666558) q[0];
ry(-0.3104867330961585) q[1];
cx q[0],q[1];
ry(-2.7570822677591) q[2];
ry(-2.0034342948586197) q[3];
cx q[2],q[3];
ry(0.622623049746339) q[2];
ry(1.3627070738169156) q[3];
cx q[2],q[3];
ry(1.8411582807466258) q[0];
ry(-1.2106996233758593) q[2];
cx q[0],q[2];
ry(2.5158793706049956) q[0];
ry(-1.3714701103872895) q[2];
cx q[0],q[2];
ry(1.902433424226384) q[1];
ry(-2.580590020835537) q[3];
cx q[1],q[3];
ry(3.074274162529887) q[1];
ry(-3.004661800251152) q[3];
cx q[1],q[3];
ry(1.4134630715741698) q[0];
ry(-0.5485596352250219) q[3];
cx q[0],q[3];
ry(0.9295571969568723) q[0];
ry(3.0830589084761613) q[3];
cx q[0],q[3];
ry(-1.1998395497864731) q[1];
ry(-2.3114376918841195) q[2];
cx q[1],q[2];
ry(2.0851885865697763) q[1];
ry(2.1207996888131495) q[2];
cx q[1],q[2];
ry(-0.035299694092356226) q[0];
ry(1.5023300578474352) q[1];
cx q[0],q[1];
ry(-0.9267624184807829) q[0];
ry(0.7002026937747496) q[1];
cx q[0],q[1];
ry(1.7841822313652567) q[2];
ry(2.1316331796348065) q[3];
cx q[2],q[3];
ry(2.616553284744745) q[2];
ry(0.16019444495602655) q[3];
cx q[2],q[3];
ry(1.9772022652822) q[0];
ry(-0.27811793394125367) q[2];
cx q[0],q[2];
ry(-0.6170265945975171) q[0];
ry(0.38817595023225326) q[2];
cx q[0],q[2];
ry(-2.03624695281703) q[1];
ry(0.6090594543387704) q[3];
cx q[1],q[3];
ry(2.239594898152479) q[1];
ry(0.7324519963583395) q[3];
cx q[1],q[3];
ry(-2.1107961252902347) q[0];
ry(-1.8334060697305619) q[3];
cx q[0],q[3];
ry(-3.0435704652118347) q[0];
ry(1.733350071490401) q[3];
cx q[0],q[3];
ry(1.7454488959112489) q[1];
ry(-2.0945426361470987) q[2];
cx q[1],q[2];
ry(1.7024879411270453) q[1];
ry(-1.8093498536427424) q[2];
cx q[1],q[2];
ry(2.9113830251235404) q[0];
ry(-1.4949713430046898) q[1];
cx q[0],q[1];
ry(-2.9197377088121774) q[0];
ry(0.915331125629204) q[1];
cx q[0],q[1];
ry(1.170129790113583) q[2];
ry(-2.2041742017955634) q[3];
cx q[2],q[3];
ry(-1.3128287217156789) q[2];
ry(1.5799142314741177) q[3];
cx q[2],q[3];
ry(-1.3510422110759384) q[0];
ry(-2.6273339623149337) q[2];
cx q[0],q[2];
ry(0.8087890715933156) q[0];
ry(-0.9857724848107996) q[2];
cx q[0],q[2];
ry(-0.4719738989734627) q[1];
ry(-0.1505905538498418) q[3];
cx q[1],q[3];
ry(1.2365565658973239) q[1];
ry(2.549656435865365) q[3];
cx q[1],q[3];
ry(-2.7335820903339374) q[0];
ry(-1.3548322854377508) q[3];
cx q[0],q[3];
ry(-2.940097735806133) q[0];
ry(1.5019285423844868) q[3];
cx q[0],q[3];
ry(-2.309968363118698) q[1];
ry(-1.8389524693968438) q[2];
cx q[1],q[2];
ry(1.6863968014113409) q[1];
ry(-1.1970630484267994) q[2];
cx q[1],q[2];
ry(-1.5245037233348013) q[0];
ry(-0.8023447137574077) q[1];
cx q[0],q[1];
ry(2.643983444514625) q[0];
ry(0.6166841956797321) q[1];
cx q[0],q[1];
ry(2.9293480131327767) q[2];
ry(1.9784550389714708) q[3];
cx q[2],q[3];
ry(-2.6970067407925327) q[2];
ry(1.6238484250536303) q[3];
cx q[2],q[3];
ry(1.4283417705718464) q[0];
ry(-0.4180632975821199) q[2];
cx q[0],q[2];
ry(-0.25463026114837106) q[0];
ry(0.6613431175120388) q[2];
cx q[0],q[2];
ry(-0.10518831606062039) q[1];
ry(-2.817613750776753) q[3];
cx q[1],q[3];
ry(2.1109827306514264) q[1];
ry(1.393047591629207) q[3];
cx q[1],q[3];
ry(-0.3144194212149207) q[0];
ry(-2.155530246418855) q[3];
cx q[0],q[3];
ry(-2.7209815790964873) q[0];
ry(2.7833321237255637) q[3];
cx q[0],q[3];
ry(-2.5149647086953526) q[1];
ry(-0.09479887206144966) q[2];
cx q[1],q[2];
ry(2.9842622013859303) q[1];
ry(1.3549304074204267) q[2];
cx q[1],q[2];
ry(2.6757088746377673) q[0];
ry(-3.0960776875856104) q[1];
ry(-1.9406054268486528) q[2];
ry(-1.6070125566696207) q[3];