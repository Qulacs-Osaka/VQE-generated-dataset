OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.355550637259177) q[0];
ry(-1.3673747445256932) q[1];
cx q[0],q[1];
ry(2.0992823165998082) q[0];
ry(2.9950469108909727) q[1];
cx q[0],q[1];
ry(-1.340535233849995) q[2];
ry(2.2399322613956807) q[3];
cx q[2],q[3];
ry(0.27104899767156504) q[2];
ry(-1.084595412441553) q[3];
cx q[2],q[3];
ry(-3.0911121728527706) q[0];
ry(-1.678289741084427) q[2];
cx q[0],q[2];
ry(-0.3104529140095549) q[0];
ry(2.1239018229085933) q[2];
cx q[0],q[2];
ry(-2.5692558664729153) q[1];
ry(-0.22063337014135254) q[3];
cx q[1],q[3];
ry(0.6798238100298075) q[1];
ry(2.382220037165474) q[3];
cx q[1],q[3];
ry(0.36858241650614343) q[0];
ry(-1.0146092258398847) q[3];
cx q[0],q[3];
ry(-0.7434621484898072) q[0];
ry(-0.41427724676852345) q[3];
cx q[0],q[3];
ry(-0.6125007595851732) q[1];
ry(0.5809646734541722) q[2];
cx q[1],q[2];
ry(1.2988282758337322) q[1];
ry(2.396315801886023) q[2];
cx q[1],q[2];
ry(-1.8676672750468144) q[0];
ry(3.1057695305544213) q[1];
cx q[0],q[1];
ry(-2.176667149363106) q[0];
ry(2.752248530412676) q[1];
cx q[0],q[1];
ry(-1.0410104816115329) q[2];
ry(0.17018654942704448) q[3];
cx q[2],q[3];
ry(1.9796116687666159) q[2];
ry(-2.0473661842722786) q[3];
cx q[2],q[3];
ry(-1.1857354902678734) q[0];
ry(-2.767350121373051) q[2];
cx q[0],q[2];
ry(0.4136255337818718) q[0];
ry(-0.046034587725231724) q[2];
cx q[0],q[2];
ry(0.7246423901730574) q[1];
ry(-0.7041840860219786) q[3];
cx q[1],q[3];
ry(0.8436315238159154) q[1];
ry(-0.22475064128199487) q[3];
cx q[1],q[3];
ry(-0.9679276817318259) q[0];
ry(-2.12201014343403) q[3];
cx q[0],q[3];
ry(3.0002171800987214) q[0];
ry(-0.6929380542054266) q[3];
cx q[0],q[3];
ry(1.4819574867899314) q[1];
ry(-1.736623637788061) q[2];
cx q[1],q[2];
ry(-2.785017532323624) q[1];
ry(3.098804952386179) q[2];
cx q[1],q[2];
ry(3.0848252004560717) q[0];
ry(-2.881575224436237) q[1];
cx q[0],q[1];
ry(1.1769451018131178) q[0];
ry(1.7196306338259404) q[1];
cx q[0],q[1];
ry(0.26140738910709727) q[2];
ry(-2.3539343709018454) q[3];
cx q[2],q[3];
ry(-1.3089435559420053) q[2];
ry(0.16661614056496343) q[3];
cx q[2],q[3];
ry(1.5532193837521102) q[0];
ry(-0.914927063544611) q[2];
cx q[0],q[2];
ry(2.7612197821231996) q[0];
ry(0.9675798751654758) q[2];
cx q[0],q[2];
ry(0.14325819488872146) q[1];
ry(0.4465541217590232) q[3];
cx q[1],q[3];
ry(0.25057906900262494) q[1];
ry(-0.9193790071257002) q[3];
cx q[1],q[3];
ry(-2.0386882732110196) q[0];
ry(2.090070574830561) q[3];
cx q[0],q[3];
ry(-0.6219641215556315) q[0];
ry(0.06257107897844277) q[3];
cx q[0],q[3];
ry(1.1903247959745642) q[1];
ry(3.1225688153355993) q[2];
cx q[1],q[2];
ry(2.0885197326629976) q[1];
ry(-1.3147123814014965) q[2];
cx q[1],q[2];
ry(2.2286570656527322) q[0];
ry(-0.9448265098714669) q[1];
cx q[0],q[1];
ry(1.5299765278042918) q[0];
ry(-1.860642841399633) q[1];
cx q[0],q[1];
ry(-1.327562233283449) q[2];
ry(-0.42398168293412564) q[3];
cx q[2],q[3];
ry(0.48289798803001993) q[2];
ry(-2.546886506632095) q[3];
cx q[2],q[3];
ry(-1.5068724125887636) q[0];
ry(-2.0247242097596603) q[2];
cx q[0],q[2];
ry(2.9345433789459534) q[0];
ry(-2.3262376281519903) q[2];
cx q[0],q[2];
ry(-0.33238915329908225) q[1];
ry(-0.5299728961467185) q[3];
cx q[1],q[3];
ry(1.0376454508257336) q[1];
ry(0.07577545615263749) q[3];
cx q[1],q[3];
ry(-2.3347981302513654) q[0];
ry(0.4521132355392865) q[3];
cx q[0],q[3];
ry(2.1465435442355485) q[0];
ry(1.5101056017856533) q[3];
cx q[0],q[3];
ry(1.8344413129254766) q[1];
ry(1.9507292254132356) q[2];
cx q[1],q[2];
ry(1.8848260515391955) q[1];
ry(-0.008172541273810391) q[2];
cx q[1],q[2];
ry(0.24908579993522414) q[0];
ry(2.161767177199094) q[1];
cx q[0],q[1];
ry(-1.1787989191735195) q[0];
ry(1.3178594563909876) q[1];
cx q[0],q[1];
ry(-0.6301789361465042) q[2];
ry(-0.803718781470449) q[3];
cx q[2],q[3];
ry(2.1732244784219517) q[2];
ry(-2.747653961429086) q[3];
cx q[2],q[3];
ry(0.9521776526225105) q[0];
ry(-2.012683841621545) q[2];
cx q[0],q[2];
ry(-2.0231482272464483) q[0];
ry(0.3464157815353646) q[2];
cx q[0],q[2];
ry(-0.7238893940148754) q[1];
ry(2.974911724244179) q[3];
cx q[1],q[3];
ry(-0.34562076854058177) q[1];
ry(1.2571075132011755) q[3];
cx q[1],q[3];
ry(-3.1101109913159672) q[0];
ry(-0.8731185166859039) q[3];
cx q[0],q[3];
ry(0.7774040975767065) q[0];
ry(1.7203984889349233) q[3];
cx q[0],q[3];
ry(0.728767370974591) q[1];
ry(-1.1444497335806556) q[2];
cx q[1],q[2];
ry(-2.999358511905215) q[1];
ry(0.7652927153348043) q[2];
cx q[1],q[2];
ry(0.6868636478926478) q[0];
ry(0.468438480644917) q[1];
cx q[0],q[1];
ry(-0.2173057212149283) q[0];
ry(-2.603284335545666) q[1];
cx q[0],q[1];
ry(-1.349825914964831) q[2];
ry(-1.2115571475108666) q[3];
cx q[2],q[3];
ry(-0.15630704777353355) q[2];
ry(1.2721946356311926) q[3];
cx q[2],q[3];
ry(-0.05795481987076654) q[0];
ry(-0.8025577447848686) q[2];
cx q[0],q[2];
ry(-1.7462440446587262) q[0];
ry(-1.0894075018776068) q[2];
cx q[0],q[2];
ry(-2.8023037304140064) q[1];
ry(-1.6369720182593632) q[3];
cx q[1],q[3];
ry(0.06272107386061528) q[1];
ry(-1.1379434881747588) q[3];
cx q[1],q[3];
ry(-0.26883858123658155) q[0];
ry(-1.5486477588180705) q[3];
cx q[0],q[3];
ry(1.0332548942026563) q[0];
ry(-2.4984744218932913) q[3];
cx q[0],q[3];
ry(2.6970898784208965) q[1];
ry(-2.127177101771311) q[2];
cx q[1],q[2];
ry(-2.935536333363179) q[1];
ry(0.7442296920588395) q[2];
cx q[1],q[2];
ry(2.0571504582708036) q[0];
ry(1.0196409637012982) q[1];
ry(0.2789769006148464) q[2];
ry(1.7574112674141) q[3];