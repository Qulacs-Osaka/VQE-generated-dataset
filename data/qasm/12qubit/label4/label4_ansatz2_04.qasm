OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-3.141592079782949) q[0];
rz(1.8595772602987728) q[0];
ry(-1.6950199886307632e-06) q[1];
rz(-1.4699621916784782) q[1];
ry(-1.0946537803101216e-07) q[2];
rz(-1.1277107330862124) q[2];
ry(4.1363029990891046e-07) q[3];
rz(0.8401950652391871) q[3];
ry(4.84180125045914e-07) q[4];
rz(-2.7048890074859657) q[4];
ry(5.105598495092067e-07) q[5];
rz(1.6644618216350198) q[5];
ry(-8.275299694240458e-07) q[6];
rz(-2.492050600755877) q[6];
ry(-5.79242161735269e-07) q[7];
rz(-0.15344107715533184) q[7];
ry(1.5707965454652886) q[8];
rz(1.5707951011090504) q[8];
ry(-1.5707965065082379) q[9];
rz(1.806109511964582) q[9];
ry(1.0227335069754498e-05) q[10];
rz(0.5164493389067065) q[10];
ry(1.1110377612499835) q[11];
rz(0.421736713508384) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.1415924243977242) q[0];
rz(-1.1630769066103424) q[0];
ry(3.1415924505475874) q[1];
rz(-2.915727480284921) q[1];
ry(3.141591858240261) q[2];
rz(-0.7299903740008826) q[2];
ry(-4.188887707812228e-07) q[3];
rz(1.4635486413303775) q[3];
ry(6.862079358072037e-06) q[4];
rz(-2.7071301984230876) q[4];
ry(1.5707855595276863) q[5];
rz(1.5708012637383415) q[5];
ry(-1.5707958760457286) q[6];
rz(-0.04885778593594824) q[6];
ry(2.350191763199316e-06) q[7];
rz(-1.328310857189897) q[7];
ry(1.3234932048147736) q[8];
rz(-0.012083874743019361) q[8];
ry(3.1415910655498265) q[9];
rz(0.3197025923576069) q[9];
ry(1.570796626729189) q[10];
rz(2.1222854111842877) q[10];
ry(-3.81952308803335e-06) q[11];
rz(-1.9925340134729606) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.5707953957313563) q[0];
rz(-1.5708248322658311) q[0];
ry(-3.0927515917414894) q[1];
rz(-1.5699436040899641) q[1];
ry(2.044238339604359) q[2];
rz(1.570805683615739) q[2];
ry(2.299165032910037e-05) q[3];
rz(-0.02401253358356853) q[3];
ry(3.141590697787988) q[4];
rz(-2.6244048163116975) q[4];
ry(1.5707954568983968) q[5];
rz(0.797072800527391) q[5];
ry(0.00016856805318938645) q[6];
rz(-2.336110208555046) q[6];
ry(1.5707976473883842) q[7];
rz(-2.6086388588487854) q[7];
ry(4.413584131057746e-06) q[8];
rz(0.8051958793063383) q[8];
ry(1.5707862197851983) q[9];
rz(2.5667989440741903) q[9];
ry(1.2020001807614025e-06) q[10];
rz(-2.347606812563611) q[10];
ry(-1.570805484890517) q[11];
rz(2.937691287241331) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5707945388514815) q[0];
rz(2.2614289989952855e-05) q[0];
ry(1.5707960716150255) q[1];
rz(-1.5749727935750975) q[1];
ry(1.5707965177688115) q[2];
rz(-3.1415853902820317) q[2];
ry(-1.5707954534355102) q[3];
rz(3.1083077235537857) q[3];
ry(3.1415921250558156) q[4];
rz(0.34492618043068196) q[4];
ry(-2.029795036051496e-06) q[5];
rz(-1.20729927453117) q[5];
ry(3.1415915293907135) q[6];
rz(-2.183589244377673) q[6];
ry(3.1415870568175586) q[7];
rz(-2.8331199916780148) q[7];
ry(-1.7358015735879917e-07) q[8];
rz(-2.378822200225629) q[8];
ry(-4.0581159951669294e-07) q[9];
rz(0.9283084205952837) q[9];
ry(-3.262939717689577e-08) q[10];
rz(-1.1514789748043706) q[10];
ry(-1.1997519129280931e-06) q[11];
rz(3.056827085030883) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5707959910932585) q[0];
rz(-1.5707957787557634) q[0];
ry(-0.4049186608159149) q[1];
rz(-1.5662655660166325) q[1];
ry(1.5708025747676209) q[2];
rz(1.521615440910012) q[2];
ry(-0.0017863117205312242) q[3];
rz(0.03328141240696484) q[3];
ry(-5.347827087831547e-07) q[4];
rz(0.769435926528108) q[4];
ry(-2.7447960349036293e-06) q[5];
rz(-2.731197024064978) q[5];
ry(3.141592642679928) q[6];
rz(0.20136235884072223) q[6];
ry(-3.1415891573257477) q[7];
rz(-0.22447746805474278) q[7];
ry(-1.2139443024139496e-06) q[8];
rz(0.816572175021193) q[8];
ry(3.1414231936166024) q[9];
rz(1.5834437299827915) q[9];
ry(-1.921273293664169e-07) q[10];
rz(2.9646050126519854) q[10];
ry(0.0001452887946493675) q[11];
rz(0.38866696791628885) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5707864796055775) q[0];
rz(0.9822643312217441) q[0];
ry(1.5707966902764927) q[1];
rz(-2.75880722698825) q[1];
ry(1.5708290539401655) q[2];
rz(-0.0979169934524705) q[2];
ry(1.5707967021118137) q[3];
rz(-1.9117969106987607) q[3];
ry(1.5707982634699507) q[4];
rz(-0.014357798094496905) q[4];
ry(1.5707852392866002) q[5];
rz(1.5707942857161596) q[5];
ry(1.5707966697438636) q[6];
rz(3.1288543047509223) q[6];
ry(-1.5707943273961442) q[7];
rz(0.9805872073888313) q[7];
ry(2.1480821030883623e-06) q[8];
rz(0.7691061284223495) q[8];
ry(-1.3653686172787616e-06) q[9];
rz(0.1078493482176593) q[9];
ry(-0.012736969801085926) q[10];
rz(-1.5929607149368152) q[10];
ry(8.259237711543124e-07) q[11];
rz(-0.09999087673772575) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-6.1042804779287e-07) q[0];
rz(-1.70591970813119) q[0];
ry(-3.1415923883249954) q[1];
rz(1.772781563819455) q[1];
ry(-4.521211817612293e-07) q[2];
rz(-2.5868303102221324) q[2];
ry(-7.403550217333077e-08) q[3];
rz(2.956971878983203) q[3];
ry(-3.1415922455768848) q[4];
rz(0.5152383887489732) q[4];
ry(1.5707981277478362) q[5];
rz(-3.141591034735243) q[5];
ry(1.5707976029150146) q[6];
rz(1.5707956404499832) q[6];
ry(3.141590913410355) q[7];
rz(0.9805779010765793) q[7];
ry(-0.06828494622935999) q[8];
rz(4.396827975838846e-05) q[8];
ry(-3.1414719619965905) q[9];
rz(-1.803832291205279) q[9];
ry(5.070449174403726e-06) q[10];
rz(0.2891927726822433) q[10];
ry(-0.3666421491741989) q[11];
rz(-1.0927763980816962e-05) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.141589199356983) q[0];
rz(-0.8109428577026945) q[0];
ry(3.141592571375681) q[1];
rz(2.250751362805035) q[1];
ry(9.83989961511611e-07) q[2];
rz(-0.5441349973691206) q[2];
ry(-8.41473530259784e-07) q[3];
rz(2.957169790356018) q[3];
ry(-1.1478368385006283e-06) q[4];
rz(-2.187681024778411) q[4];
ry(-1.5707961112327244) q[5];
rz(-2.280795707130172) q[5];
ry(1.5708038707199943) q[6];
rz(3.054303566168468) q[6];
ry(-1.570799139102828) q[7];
rz(2.4316115465667805) q[7];
ry(1.5707853742542686) q[8];
rz(-0.08726455741506722) q[8];
ry(-1.5707962378290574) q[9];
rz(0.860815281048321) q[9];
ry(-3.141592270027255) q[10];
rz(0.19677493274440755) q[10];
ry(-1.5707958241758195) q[11];
rz(2.431613197967328) q[11];