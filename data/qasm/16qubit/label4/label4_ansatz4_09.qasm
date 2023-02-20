OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(3.014574637454509) q[0];
rz(0.04653708895756994) q[0];
ry(-0.47547436241015706) q[1];
rz(1.6090827576607962) q[1];
ry(0.07727457240332261) q[2];
rz(1.24430849146109) q[2];
ry(-0.0002478037412902978) q[3];
rz(1.360169720650986) q[3];
ry(1.4823874722965629) q[4];
rz(1.572479657472602) q[4];
ry(1.5688667527020719) q[5];
rz(1.6417964139461285) q[5];
ry(3.141577305230171) q[6];
rz(-2.426815062182827) q[6];
ry(9.525298342616394e-05) q[7];
rz(0.17494121561900983) q[7];
ry(-0.174952645807557) q[8];
rz(-2.509270612667551) q[8];
ry(-1.5778640251410965) q[9];
rz(-3.1414415685578367) q[9];
ry(-0.16016634699836144) q[10];
rz(0.2853621283360071) q[10];
ry(0.05228193868398322) q[11];
rz(0.2102781812070647) q[11];
ry(1.573011196989186) q[12];
rz(-3.1412713805447035) q[12];
ry(0.06157998253130259) q[13];
rz(-2.6768816372772357) q[13];
ry(-1.3693937841082766) q[14];
rz(0.06209259725699478) q[14];
ry(-1.5723114914821963) q[15];
rz(-3.1385611544347918) q[15];
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
ry(-3.1354905588926942) q[0];
rz(-1.5735357721477747) q[0];
ry(1.635968572915667) q[1];
rz(-1.6748475945514194) q[1];
ry(0.04458689203174693) q[2];
rz(1.4916298446872924) q[2];
ry(0.006094377975365717) q[3];
rz(1.3191381372453872) q[3];
ry(1.6396924727862094) q[4];
rz(-1.5486784682795365) q[4];
ry(1.5479207128573105) q[5];
rz(-0.024053908633884017) q[5];
ry(-1.6941874205513896) q[6];
rz(1.5609044160624936) q[6];
ry(1.570456908036662) q[7];
rz(3.141478433997295) q[7];
ry(3.1413098632874976) q[8];
rz(0.3943851292969107) q[8];
ry(1.5776298101333346) q[9];
rz(0.5587029384245312) q[9];
ry(-0.005526240578044295) q[10];
rz(1.303696607753678) q[10];
ry(2.7310057442420024) q[11];
rz(-2.5226522435060548) q[11];
ry(-2.533677133796466) q[12];
rz(3.141117406861983) q[12];
ry(-0.033560569476358854) q[13];
rz(-0.37370477307671557) q[13];
ry(0.9933703975352306) q[14];
rz(-3.109240377812288) q[14];
ry(-2.14135644170329) q[15];
rz(-1.5711949645330003) q[15];
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
ry(1.5947844851351753) q[0];
rz(0.008859857503979995) q[0];
ry(3.015101823620896) q[1];
rz(2.235509750014209) q[1];
ry(3.1270182343849524) q[2];
rz(-1.7835458152285746) q[2];
ry(-1.562233127482098) q[3];
rz(1.5570876222116468) q[3];
ry(1.574004949092747) q[4];
rz(-0.0753344942093195) q[4];
ry(1.573283804335713) q[5];
rz(-0.0065650732584548845) q[5];
ry(1.5706411218923442) q[6];
rz(0.009944184836808924) q[6];
ry(-1.559892862050149) q[7];
rz(-1.3449890187447269) q[7];
ry(0.37645468331443543) q[8];
rz(-0.5396238804542408) q[8];
ry(0.08981701379255123) q[9];
rz(-0.8778661490417576) q[9];
ry(-3.1413052786586517) q[10];
rz(-1.514272926794919) q[10];
ry(0.00017456475077791119) q[11];
rz(1.2793367148332053) q[11];
ry(1.570301105649224) q[12];
rz(-1.8259474084270528) q[12];
ry(3.028248575467992) q[13];
rz(-1.3961908215485612) q[13];
ry(-0.5431004986861391) q[14];
rz(-2.9088694209587986) q[14];
ry(-1.8916136765192197) q[15];
rz(3.1307621543594264) q[15];
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
ry(1.5729517968757383) q[0];
rz(-3.112447734037791) q[0];
ry(1.5783727580555502) q[1];
rz(-1.5783214189966568) q[1];
ry(-3.138852522749753) q[2];
rz(-1.6795713525085332) q[2];
ry(0.6527473884187706) q[3];
rz(-3.1227694114956868) q[3];
ry(-0.12393927136974181) q[4];
rz(0.9775278093675378) q[4];
ry(-0.583408621667214) q[5];
rz(-1.440516892718168) q[5];
ry(-1.5710716234064763) q[6];
rz(-0.30861707748803013) q[6];
ry(0.09409345754456659) q[7];
rz(2.924518399036428) q[7];
ry(-1.0155963150162339e-05) q[8];
rz(0.8809414215606727) q[8];
ry(3.1414704358748198) q[9];
rz(0.45909363904869593) q[9];
ry(1.5699761283881895) q[10];
rz(0.3637607456628782) q[10];
ry(-1.5710428566327825) q[11];
rz(-2.077286298415176) q[11];
ry(0.6097426138562563) q[12];
rz(3.1351305408916543) q[12];
ry(0.00024266497331115747) q[13];
rz(-0.622500061325054) q[13];
ry(-1.4735193923469636) q[14];
rz(0.00010402414025966866) q[14];
ry(2.2089900344694557) q[15];
rz(2.6341336920224787) q[15];
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
ry(-1.5947800612642162) q[0];
rz(1.5524622815676437) q[0];
ry(1.5673076245739745) q[1];
rz(3.0751041388791602) q[1];
ry(0.010668641212211227) q[2];
rz(2.9343229626905027) q[2];
ry(1.5740564021944774) q[3];
rz(-0.5899415846167254) q[3];
ry(3.1322523098202013) q[4];
rz(-0.6657817228678196) q[4];
ry(-0.00753883420554427) q[5];
rz(-1.6867001693279677) q[5];
ry(-1.5713063386462647) q[6];
rz(1.571804641804924) q[6];
ry(-1.9031816002054907) q[7];
rz(1.565009354802637) q[7];
ry(1.5705334263977133) q[8];
rz(-3.135957387367327) q[8];
ry(-3.141558384324099) q[9];
rz(2.6980033681934157) q[9];
ry(0.0011369166029524536) q[10];
rz(-0.3644182057835654) q[10];
ry(-0.00030035307826103974) q[11];
rz(-3.065016389248558) q[11];
ry(0.0019775662676400074) q[12];
rz(-1.2733194174709135) q[12];
ry(0.0007254109345760667) q[13];
rz(2.3397340366953596) q[13];
ry(-2.0521704329021357) q[14];
rz(1.2599092161987135) q[14];
ry(0.0286188227754214) q[15];
rz(0.4981866726273365) q[15];
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
ry(-1.570370354097967) q[0];
rz(3.113312971126754) q[0];
ry(1.5687716802224339) q[1];
rz(1.7912621861629716) q[1];
ry(0.0578570917485246) q[2];
rz(-1.5333134577406582) q[2];
ry(-3.0698033863684278) q[3];
rz(-2.159779433726425) q[3];
ry(1.7150816695075979) q[4];
rz(-0.009481577219230708) q[4];
ry(-1.5814747362193584) q[5];
rz(-3.098600796860102) q[5];
ry(1.570647742765546) q[6];
rz(3.1411330425975006) q[6];
ry(0.03756843191559511) q[7];
rz(0.004559687162911884) q[7];
ry(-0.047484571263955166) q[8];
rz(-1.576823752594188) q[8];
ry(-3.1415819914088248) q[9];
rz(0.33087775394081653) q[9];
ry(2.2643344937814818) q[10];
rz(-1.5727057023189426) q[10];
ry(0.0022589505581604996) q[11];
rz(-2.241098582173863) q[11];
ry(0.09738668136370786) q[12];
rz(2.952305173094334) q[12];
ry(3.1415120416773323) q[13];
rz(0.4883637587098084) q[13];
ry(-0.2892156184396635) q[14];
rz(0.32666540689747375) q[14];
ry(2.2083479087074576) q[15];
rz(1.5624535460216902) q[15];
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
ry(-1.2567711252985811) q[0];
rz(0.06125759861415857) q[0];
ry(0.03859317842952219) q[1];
rz(2.1175953677566652) q[1];
ry(1.5713760364451037) q[2];
rz(-2.5433644365926362) q[2];
ry(2.3891425421270855) q[3];
rz(-1.5701018942728222) q[3];
ry(-1.7914182635969667e-05) q[4];
rz(2.2489025411057244) q[4];
ry(3.141572940215622) q[5];
rz(1.90162164676605) q[5];
ry(-1.565978660460754) q[6];
rz(-3.141418559271717) q[6];
ry(0.28908144117378903) q[7];
rz(1.3403901132595515) q[7];
ry(-1.5708521970910616) q[8];
rz(2.078437127626164) q[8];
ry(-0.30872215039272144) q[9];
rz(3.1329093610777634) q[9];
ry(1.5711032668398355) q[10];
rz(0.7362039476061274) q[10];
ry(-3.141048208337765) q[11];
rz(2.050183274213618) q[11];
ry(-1.5739150328236788) q[12];
rz(-1.5797546074758504) q[12];
ry(-3.141570785839488) q[13];
rz(-0.17759500855635413) q[13];
ry(-1.5711943713246181) q[14];
rz(0.8145254238729003) q[14];
ry(1.699171241431577) q[15];
rz(-1.5455052043912498) q[15];
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
ry(-0.07032584120525165) q[0];
rz(2.207202046497047) q[0];
ry(-2.9015958873833454) q[1];
rz(-1.3935711846692842) q[1];
ry(0.00032425578207728734) q[2];
rz(-2.5044144506964336) q[2];
ry(-2.7935629597519496) q[3];
rz(1.5758137322314958) q[3];
ry(-3.133594995903709) q[4];
rz(0.6802482802231111) q[4];
ry(-3.1397881135130423) q[5];
rz(1.7561800518875499) q[5];
ry(1.5704077474189146) q[6];
rz(-1.5640711763685138) q[6];
ry(-3.131210039693709) q[7];
rz(-3.0624619251118834) q[7];
ry(-0.00015902163902214994) q[8];
rz(-2.0777968349566285) q[8];
ry(-3.100321746364646) q[9];
rz(-1.5689807594255756) q[9];
ry(3.1408293586125695) q[10];
rz(-0.7778338576475985) q[10];
ry(3.121639145676279) q[11];
rz(0.7167378321925293) q[11];
ry(1.5717490015826812) q[12];
rz(2.2602960059072483) q[12];
ry(-2.9654606156448438e-05) q[13];
rz(2.2961088832047762) q[13];
ry(3.140817608109941) q[14];
rz(0.596156859762484) q[14];
ry(1.570851503787658) q[15];
rz(-1.5463849346893959) q[15];
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
ry(-0.04336746366729205) q[0];
rz(0.8769834603536332) q[0];
ry(3.1295471723506467) q[1];
rz(-2.185966554905283) q[1];
ry(-0.0010155887932929986) q[2];
rz(-1.235309374068475) q[2];
ry(2.3080881367064956) q[3];
rz(-3.1384535825504654) q[3];
ry(-2.7658206448633114) q[4];
rz(1.5709517437210536) q[4];
ry(3.1415554635207616) q[5];
rz(-0.29541958044679545) q[5];
ry(-2.6281554844503057) q[6];
rz(0.05111063947357942) q[6];
ry(-3.1115248561814473) q[7];
rz(-1.2007415102726777) q[7];
ry(1.5714765970745868) q[8];
rz(1.5879636598553466) q[8];
ry(1.0799350466112057) q[9];
rz(2.7638539727782954) q[9];
ry(3.139935199657553) q[10];
rz(2.780225540725229) q[10];
ry(-3.1380541890730558) q[11];
rz(2.2791580314038096) q[11];
ry(0.00018446048238018875) q[12];
rz(1.0283031814779033) q[12];
ry(-1.5701924105282508) q[13];
rz(-3.1401719766231104) q[13];
ry(-1.864268028354851) q[14];
rz(3.1143119602195464) q[14];
ry(3.141141978129665) q[15];
rz(1.5838200887090155) q[15];
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
ry(-1.609560552186855) q[0];
rz(-1.748791023990763) q[0];
ry(1.77160800380396) q[1];
rz(-1.4998079894936494) q[1];
ry(1.5713754788206535) q[2];
rz(3.139640099304525) q[2];
ry(1.571447150422579) q[3];
rz(-2.8987787491300576) q[3];
ry(0.11028778927992831) q[4];
rz(1.5701688741908104) q[4];
ry(-3.141324290053662) q[5];
rz(0.7094209805716988) q[5];
ry(-3.1412451816889546) q[6];
rz(-3.097962776556094) q[6];
ry(7.997097242373741e-05) q[7];
rz(1.4808905675981763) q[7];
ry(0.05137993973583296) q[8];
rz(1.6488561487994393) q[8];
ry(0.0003939045296066368) q[9];
rz(-1.1949704480448355) q[9];
ry(-3.1415390016246247) q[10];
rz(1.1504179992756391) q[10];
ry(-2.7320412299627046) q[11];
rz(3.0938885777318065e-05) q[11];
ry(-0.05995068298499857) q[12];
rz(-0.1456592722464478) q[12];
ry(1.57058311031714) q[13];
rz(0.00014949079845827384) q[13];
ry(-2.7455218070590632) q[14];
rz(-0.33405221059320583) q[14];
ry(-1.5706129369746948) q[15];
rz(2.7036253594297093) q[15];
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
ry(3.1233525451391717) q[0];
rz(-1.9941169155621754) q[0];
ry(-2.7150866959775017) q[1];
rz(-1.6282912758095174) q[1];
ry(-2.2667647455306703) q[2];
rz(-1.5763674935692382) q[2];
ry(0.0001679589187488295) q[3];
rz(-1.4046040137477707) q[3];
ry(1.5711351140506502) q[4];
rz(1.2567573889659909) q[4];
ry(0.0004486709847064673) q[5];
rz(-2.159841508889704) q[5];
ry(-2.0155869268737696) q[6];
rz(2.219934284125478) q[6];
ry(-0.3337509866033803) q[7];
rz(-3.1415264023702214) q[7];
ry(3.1413653972265356) q[8];
rz(-1.4611360628014294) q[8];
ry(-1.5717304601969797) q[9];
rz(-2.151509494051391) q[9];
ry(-1.5647340286074263) q[10];
rz(-1.5697831178422523) q[10];
ry(-1.5704689958986462) q[11];
rz(-1.5708124149094969) q[11];
ry(1.5707545442522521) q[12];
rz(-1.547319639555577) q[12];
ry(-1.5707346794375028) q[13];
rz(3.14075558863894) q[13];
ry(3.136451796386082) q[14];
rz(-0.300969664256773) q[14];
ry(-0.004373255181548295) q[15];
rz(-2.70365399747647) q[15];
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
ry(-3.1294208620869095) q[0];
rz(2.8970514545549886) q[0];
ry(-1.7489432007966403) q[1];
rz(0.0377256802719259) q[1];
ry(2.687033678136604) q[2];
rz(3.1407048127633512) q[2];
ry(3.141522950268781) q[3];
rz(-0.3949977620354907) q[3];
ry(0.0012648007174194118) q[4];
rz(-1.7702213390575308) q[4];
ry(0.22442585350151123) q[5];
rz(1.2745861700654766) q[5];
ry(3.1411213228123254) q[6];
rz(0.6479371521616608) q[6];
ry(-0.019363924107214204) q[7];
rz(-1.5430926012408628) q[7];
ry(0.01241667033254739) q[8];
rz(-1.5858165337047516) q[8];
ry(-3.2249877032874956e-05) q[9];
rz(1.4119527434138337) q[9];
ry(-1.5708122863917924) q[10];
rz(-8.917662384533287e-06) q[10];
ry(1.5708051081759273) q[11];
rz(-1.1256338844805303) q[11];
ry(3.1194868741042767) q[12];
rz(0.023673384463182064) q[12];
ry(1.5667368436804183) q[13];
rz(-3.0291206381887337) q[13];
ry(-1.5707605513923113) q[14];
rz(1.062739892402464) q[14];
ry(1.5705825294203724) q[15];
rz(-0.004502171302502644) q[15];
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
ry(-1.5719500336041485) q[0];
rz(1.4153572169088506) q[0];
ry(-1.5968913286458406) q[1];
rz(-3.1288365333334953) q[1];
ry(-1.5705860939141356) q[2];
rz(-1.7416849795789424) q[2];
ry(3.141166923855041) q[3];
rz(0.7763126907112677) q[3];
ry(-3.1409743922283564) q[4];
rz(-2.275680451355746) q[4];
ry(-1.5651712189593239) q[5];
rz(-3.09323953034823) q[5];
ry(-1.570489508136907) q[6];
rz(-0.25542854617242705) q[6];
ry(1.571555665204992) q[7];
rz(-1.7668336590663918) q[7];
ry(1.569801735393968) q[8];
rz(2.9271342429426683) q[8];
ry(-0.08800315307154212) q[9];
rz(0.38098009639424196) q[9];
ry(-1.570632323696345) q[10];
rz(-0.1767856630261342) q[10];
ry(0.0010406800432899552) q[11];
rz(2.104924268799442) q[11];
ry(-1.5710170373163095) q[12];
rz(-1.743691703175879) q[12];
ry(-0.020622253576971335) q[13];
rz(-2.0731353082237165) q[13];
ry(-3.141549034211068) q[14];
rz(2.462689043030717) q[14];
ry(-1.5704791625374102) q[15];
rz(-1.942399352302738) q[15];