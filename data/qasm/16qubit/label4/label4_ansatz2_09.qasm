OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.570781836949557) q[0];
rz(-3.1415553098617135) q[0];
ry(0.0002854411269472834) q[1];
rz(1.0370940011299732) q[1];
ry(2.4589578983416573) q[2];
rz(2.994188585578519) q[2];
ry(-7.491784592563323e-05) q[3];
rz(2.4633064660631616) q[3];
ry(-1.5707732353386274) q[4];
rz(3.1415745881120967) q[4];
ry(1.5707395480093782) q[5];
rz(-0.49048164714949943) q[5];
ry(-1.6020415383299573) q[6];
rz(0.07498715013976476) q[6];
ry(0.9416424287000495) q[7];
rz(-2.3392197530537806) q[7];
ry(-3.1410925255610826) q[8];
rz(-0.7966162659465141) q[8];
ry(-5.460013660071816e-05) q[9];
rz(-1.4823048279176663) q[9];
ry(1.5633569703172288) q[10];
rz(2.8250161663125906) q[10];
ry(-1.5708304875940964) q[11];
rz(2.7127799836198383) q[11];
ry(-1.5707869614163297) q[12];
rz(1.570765846767807) q[12];
ry(-3.0908493689230085) q[13];
rz(1.5693482549977036) q[13];
ry(3.6601120689129514e-07) q[14];
rz(0.16911899175716627) q[14];
ry(-3.1415905841450114) q[15];
rz(1.7750584926537147) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-2.4853153388886637) q[0];
rz(2.5906005941161525) q[0];
ry(-4.714788174330172e-06) q[1];
rz(1.8755755521219202) q[1];
ry(-5.884378593279393e-07) q[2];
rz(-1.4296890586948272) q[2];
ry(3.1415923849793166) q[3];
rz(0.41420950831559106) q[3];
ry(-0.38823721670537026) q[4];
rz(-1.5715889436759325) q[4];
ry(3.141587150971909) q[5];
rz(0.3661597028827367) q[5];
ry(-1.632958357600245e-05) q[6];
rz(1.495658686576342) q[6];
ry(3.1415824475130076) q[7];
rz(2.3842015149654774) q[7];
ry(-5.791573001661096e-06) q[8];
rz(2.0199156971127246) q[8];
ry(0.0002880189118786511) q[9];
rz(2.470130245854233) q[9];
ry(0.00015068094699000056) q[10];
rz(2.159455464041213) q[10];
ry(3.1415789925386792) q[11];
rz(1.1419760199354465) q[11];
ry(-1.5707748589908528) q[12];
rz(0.4593430335127327) q[12];
ry(-1.570785540500663) q[13];
rz(-3.1164936822938114) q[13];
ry(1.5708322471995009) q[14];
rz(-1.292367821906657) q[14];
ry(-1.5707956292642316) q[15];
rz(-1.1006139188741983e-06) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.932446662524569e-05) q[0];
rz(0.5509551764758154) q[0];
ry(-3.1401268881424738) q[1];
rz(-1.5681220824998805) q[1];
ry(3.117666608790529) q[2];
rz(-0.0063315314623428876) q[2];
ry(-7.149096539806381e-06) q[3];
rz(3.0397466914066396) q[3];
ry(-2.8939617586000668) q[4];
rz(-1.5715066308538148) q[4];
ry(-2.3607505349903253) q[5];
rz(-1.5210091484524098) q[5];
ry(1.4545003649346868) q[6];
rz(-3.092487749150175) q[6];
ry(-3.1337223856647958) q[7];
rz(0.011111202177425561) q[7];
ry(-3.1414131268470373) q[8];
rz(1.2091482233856707) q[8];
ry(3.1415919391914517) q[9];
rz(2.165051682260489) q[9];
ry(3.1415324840612686) q[10];
rz(-2.869936177730038) q[10];
ry(-0.47865008891266714) q[11];
rz(-1.5706533656374626) q[11];
ry(6.0444967315831386e-06) q[12];
rz(1.6277179932031132) q[12];
ry(-0.0001436532863578068) q[13];
rz(-1.5919673854644236) q[13];
ry(-1.7732185982875496e-06) q[14];
rz(-1.0801054749681658) q[14];
ry(1.5706530431007966) q[15];
rz(-1.5178956821534728) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-1.806328774884454) q[0];
rz(1.4858091443840638) q[0];
ry(1.5706883817524904) q[1];
rz(3.141502665414021) q[1];
ry(1.568147805867867) q[2];
rz(-0.5062336313724503) q[2];
ry(3.141520986180062) q[3];
rz(-1.4508471447387858) q[3];
ry(-2.4294032932051155) q[4];
rz(1.570891850568348) q[4];
ry(4.746858287635589e-05) q[5];
rz(-2.1792004642174376) q[5];
ry(2.8924252907609276) q[6];
rz(1.9158396047038861) q[6];
ry(-1.579621725843285) q[7];
rz(-3.1326417700422646) q[7];
ry(-2.7381845096646718) q[8];
rz(1.9866982235834807) q[8];
ry(1.5708013238549468) q[9];
rz(-0.00010669466135393436) q[9];
ry(-1.5790502823450732) q[10];
rz(1.3420686388964533) q[10];
ry(-2.8222545946353814) q[11];
rz(-1.5706275963816598) q[11];
ry(5.160029717288239e-06) q[12];
rz(-2.0507244130399602) q[12];
ry(0.00047010585453339317) q[13];
rz(-2.5824546754883446) q[13];
ry(4.27145825784007e-06) q[14];
rz(0.8016774257315485) q[14];
ry(-3.141128320781107) q[15];
rz(-2.6939712257334674) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.141163389562829) q[0];
rz(1.4858278461492591) q[0];
ry(2.8631575010100856) q[1];
rz(-1.278374329068406) q[1];
ry(3.141583922676825) q[2];
rz(-3.058051826379922) q[2];
ry(3.141591346528752) q[3];
rz(0.6695004314291525) q[3];
ry(-1.795884664618185) q[4];
rz(-0.07035786954284617) q[4];
ry(0.0007519877570804829) q[5];
rz(2.4186989124225025) q[5];
ry(3.1415854355318515) q[6];
rz(-2.115468992793799) q[6];
ry(3.1415908785176456) q[7];
rz(0.5066456846445941) q[7];
ry(-3.1415874962124977) q[8];
rz(-2.724330237680434) q[8];
ry(-1.8499974742245113) q[9];
rz(-3.141590443732151) q[9];
ry(9.611855213975673e-07) q[10];
rz(-1.9581352017403244) q[10];
ry(0.22553086663457703) q[11];
rz(2.190443699901744) q[11];
ry(1.0863113971026905e-05) q[12];
rz(-2.1329399658458796) q[12];
ry(-3.141587669034717) q[13];
rz(2.116160612516146) q[13];
ry(-1.7900160481387273) q[14];
rz(2.44572277905326) q[14];
ry(-1.36005546730611e-05) q[15];
rz(-0.3947098399491912) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(1.5708222633078308) q[0];
rz(0.5488742388684277) q[0];
ry(-3.1415782313672596) q[1];
rz(-1.3174766659883215) q[1];
ry(5.591516172845896e-06) q[2];
rz(-2.192343135214185) q[2];
ry(-2.892997930104002e-07) q[3];
rz(-0.21040089959571343) q[3];
ry(-3.664477021381174e-06) q[4];
rz(2.2835593648774934) q[4];
ry(1.3639567484347028e-06) q[5];
rz(0.7153306304202341) q[5];
ry(3.1415764558376367) q[6];
rz(-2.363062894677668) q[6];
ry(-6.141733225994983e-06) q[7];
rz(-1.1570409456920825) q[7];
ry(1.5707866781128397) q[8];
rz(1.570765228676395) q[8];
ry(1.5707917188670484) q[9];
rz(1.5707534257885083) q[9];
ry(-1.5713077419730466) q[10];
rz(-1.5708581262352654) q[10];
ry(-3.141539124427164) q[11];
rz(0.8482850876302115) q[11];
ry(1.5707935893135971) q[12];
rz(-1.4043117955714148) q[12];
ry(-0.7619251601280661) q[13];
rz(2.8728429290468607) q[13];
ry(1.3875681812707962e-08) q[14];
rz(0.9416203158896221) q[14];
ry(-1.5539524941546845) q[15];
rz(-1.5333696127704535) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.141387861415143) q[0];
rz(-1.6556594316792614) q[0];
ry(-8.511792107590566e-06) q[1];
rz(-3.093014556470447) q[1];
ry(6.979291899966711e-08) q[2];
rz(-1.5390271203501993) q[2];
ry(-9.534873773808415e-07) q[3];
rz(-1.371780589177149) q[3];
ry(3.141337576497243) q[4];
rz(-0.9717266531604618) q[4];
ry(-0.00197739626134424) q[5];
rz(1.6512085418455982) q[5];
ry(-3.1415770363494615) q[6];
rz(0.057032212197394294) q[6];
ry(3.1415903770201474) q[7];
rz(-1.5558428959944746) q[7];
ry(1.5707743930279365) q[8];
rz(0.34244745034888524) q[8];
ry(1.5707932269074836) q[9];
rz(3.141584703806083) q[9];
ry(-1.5708024921421595) q[10];
rz(-2.7220682051607024) q[10];
ry(0.010494298974339067) q[11];
rz(-0.328448225200086) q[11];
ry(3.1415807366010022) q[12];
rz(-2.747409690153299) q[12];
ry(-3.1415699360278824) q[13];
rz(-1.9751655439634863) q[13];
ry(6.5364941876550176e-06) q[14];
rz(-0.24575539704912316) q[14];
ry(-9.931649957550803e-06) q[15];
rz(1.5334205200650504) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.0794549391726176) q[0];
rz(-0.6346462266553088) q[0];
ry(1.5708028417248372) q[1];
rz(-3.1358796343796547) q[1];
ry(-1.5707919979769782) q[2];
rz(1.570888409111065) q[2];
ry(-3.141591207092462) q[3];
rz(-0.7959731040816916) q[3];
ry(0.6895782347016679) q[4];
rz(0.03599253229789312) q[4];
ry(-1.5652855763570963) q[5];
rz(-0.00042624421490737195) q[5];
ry(5.3231802513487025e-05) q[6];
rz(3.1314249964429393) q[6];
ry(-1.157464716694534e-05) q[7];
rz(0.8935941112939405) q[7];
ry(5.890033112798676e-06) q[8];
rz(0.14812792678766765) q[8];
ry(-1.5710306094908877) q[9];
rz(-1.5971600621068163) q[9];
ry(8.835223002812143e-05) q[10];
rz(1.843258849981111) q[10];
ry(0.0009081457006401956) q[11];
rz(1.670626907253074) q[11];
ry(3.1413646571724367) q[12];
rz(0.23508028026110403) q[12];
ry(-5.264681997019238e-06) q[13];
rz(2.8812337659958036) q[13];
ry(1.5705697356253454) q[14];
rz(1.8785697276044315) q[14];
ry(1.570792300476579) q[15];
rz(-0.11931491758584477) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-1.0201832123352115) q[0];
rz(-3.141450316248965) q[0];
ry(1.5707507902226685) q[1];
rz(-0.26598535528520806) q[1];
ry(1.5709416425345606) q[2];
rz(-3.141582133954182) q[2];
ry(2.0482773620214865e-06) q[3];
rz(-2.175581310263035) q[3];
ry(3.1143813061277026) q[4];
rz(3.1213265227279283) q[4];
ry(1.57076983965933) q[5];
rz(1.9741918553835003) q[5];
ry(1.5712176313143116) q[6];
rz(3.098471869150702) q[6];
ry(3.089008814635782) q[7];
rz(-1.5736113218725787) q[7];
ry(-0.003129913704381244) q[8];
rz(0.16954750509231484) q[8];
ry(3.141588805112834) q[9];
rz(-1.5253162158613023) q[9];
ry(-4.613690423345156e-06) q[10];
rz(1.5920070384826417) q[10];
ry(2.677309188272395) q[11];
rz(-1.1312793086381332) q[11];
ry(-6.738307964937462e-07) q[12];
rz(1.71447590583103) q[12];
ry(-1.3194265688909468e-06) q[13];
rz(-2.7860437599893437) q[13];
ry(-3.1415913910838555) q[14];
rz(0.01140747375319773) q[14];
ry(1.7302915504835426e-05) q[15];
rz(1.0138281530339235) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(0.5178671143774132) q[0];
rz(3.1414267176691184) q[0];
ry(3.084627882530035) q[1];
rz(0.7454868269364567) q[1];
ry(-1.5170444435618569) q[2];
rz(-0.011691157672269798) q[2];
ry(2.9326307396831985e-06) q[3];
rz(-1.5586514928657316) q[3];
ry(-2.687462687526358) q[4];
rz(-0.00010401826784089964) q[4];
ry(-0.3642442594873438) q[5];
rz(1.5567768131602797) q[5];
ry(-1.5707451545549782) q[6];
rz(1.5785609880461724) q[6];
ry(1.570764845068579) q[7];
rz(-0.8694133812537163) q[7];
ry(3.1413913203709516) q[8];
rz(-3.088186668956709) q[8];
ry(0.008897751987085695) q[9];
rz(-0.6999781799328824) q[9];
ry(3.1413880662145655) q[10];
rz(-0.8429398021060655) q[10];
ry(-2.165705930122559e-05) q[11];
rz(2.3100178532212152) q[11];
ry(-8.493550076593692e-07) q[12];
rz(1.810863588843321) q[12];
ry(3.1415579144981396) q[13];
rz(1.5603083119861116) q[13];
ry(-3.1415843113556) q[14];
rz(-1.266740145601339) q[14];
ry(-4.428803818612437e-05) q[15];
rz(-2.044439257349843) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-1.5707476525097042) q[0];
rz(-1.4993774928968748) q[0];
ry(-1.5716303956837347) q[1];
rz(-0.0001514013441248352) q[1];
ry(-0.07926160234107918) q[2];
rz(-1.5590614793851638) q[2];
ry(1.3593573573444928e-06) q[3];
rz(-2.9529314659627945) q[3];
ry(1.570782631888066) q[4];
rz(-0.6108302080040675) q[4];
ry(3.1415771120760287) q[5];
rz(0.37270514365015467) q[5];
ry(-3.1305923692356927) q[6];
rz(1.5786804581110396) q[6];
ry(-3.141556065240018) q[7];
rz(1.47368997701952) q[7];
ry(3.1415602186309504) q[8];
rz(2.1899524877095047) q[8];
ry(-0.000734501840087276) q[9];
rz(-1.1413521661628394) q[9];
ry(-1.9441013253462813e-05) q[10];
rz(3.0178614515044813) q[10];
ry(-3.141590374848142) q[11];
rz(2.3935531694115073) q[11];
ry(-7.99926234330268e-06) q[12];
rz(-1.0599730613353024) q[12];
ry(-1.3110367993895977e-05) q[13];
rz(1.464700735629036) q[13];
ry(3.1415894424780237) q[14];
rz(-3.124957352269154) q[14];
ry(2.590343047270853e-05) q[15];
rz(1.1217185533564116) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.224507108572669e-05) q[0];
rz(2.1927491835689468) q[0];
ry(-2.9714827866517965) q[1];
rz(2.5642179544696484) q[1];
ry(-1.5804938420603063) q[2];
rz(-0.7756256529233818) q[2];
ry(1.570783000169989) q[3];
rz(3.1415916509133375) q[3];
ry(-3.141537860422407) q[4];
rz(1.4420024571376413) q[4];
ry(-1.5708162301111386) q[5];
rz(1.4945700301268792) q[5];
ry(-1.6185197651153593) q[6];
rz(-0.8470910402123377) q[6];
ry(-3.141526807027296) q[7];
rz(-0.5073513255313218) q[7];
ry(-1.477686148703245e-05) q[8];
rz(-1.538754329768179) q[8];
ry(-3.1366719683885806) q[9];
rz(1.393280726183351) q[9];
ry(3.139734897958494) q[10];
rz(1.5528500785917978) q[10];
ry(-6.15126654233555e-06) q[11];
rz(-1.2098863888972637) q[11];
ry(-1.3479504552549315e-06) q[12];
rz(0.6688526418987282) q[12];
ry(3.8412477473492856e-05) q[13];
rz(0.10048335851792957) q[13];
ry(-3.141588343589088) q[14];
rz(-0.5838244371864836) q[14];
ry(3.1415585982562515) q[15];
rz(-0.02820692308439199) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.141586355554092) q[0];
rz(-1.8962480565535598) q[0];
ry(1.6913004596830203e-07) q[1];
rz(0.15928920907110733) q[1];
ry(3.1415748769947927) q[2];
rz(-0.22360345388877914) q[2];
ry(1.5707955476208044) q[3];
rz(2.723669433702465) q[3];
ry(7.902516037106011e-07) q[4];
rz(1.6406888740827155) q[4];
ry(1.0342155487563787e-06) q[5];
rz(2.799732284457979) q[5];
ry(1.463830392900434e-06) q[6];
rz(1.3989921385978743) q[6];
ry(3.416578837421269e-06) q[7];
rz(0.8615860357783537) q[7];
ry(-8.900252031791577e-07) q[8];
rz(2.4353140934928232) q[8];
ry(-2.2406817346495883e-06) q[9];
rz(1.1315493955787121) q[9];
ry(1.6574195117882607e-06) q[10];
rz(2.0313957828290925) q[10];
ry(3.1291720979642434) q[11];
rz(2.7284948219787237) q[11];
ry(1.5707905022434956) q[12];
rz(1.1963200329072456) q[12];
ry(1.5707981947638219) q[13];
rz(-1.9888102335670852) q[13];
ry(1.570797271383137) q[14];
rz(2.7671176258723436) q[14];
ry(-1.5709184733900114) q[15];
rz(-0.4180161035691308) q[15];