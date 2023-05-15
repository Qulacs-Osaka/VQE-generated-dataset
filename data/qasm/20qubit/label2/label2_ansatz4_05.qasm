OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.569329418275578) q[0];
rz(-1.2033667683520601) q[0];
ry(-1.5724378325431894) q[1];
rz(0.11910954996645541) q[1];
ry(3.1415917703767597) q[2];
rz(-1.1093704929109682) q[2];
ry(3.1415577052223713) q[3];
rz(1.2921461156834724) q[3];
ry(3.094789515190373) q[4];
rz(-0.9793660931254582) q[4];
ry(0.0008783647814859609) q[5];
rz(0.6522658944597867) q[5];
ry(-1.5707512901835865) q[6];
rz(1.574930668226771) q[6];
ry(1.5707420646426027) q[7];
rz(1.5661250402300786) q[7];
ry(0.004456129260742449) q[8];
rz(1.5159182959030675) q[8];
ry(-3.094580606502435) q[9];
rz(1.9286650683862758) q[9];
ry(-3.111495923676735) q[10];
rz(0.8166514302241117) q[10];
ry(-0.038026836448071755) q[11];
rz(0.748342081045152) q[11];
ry(-1.5709339661270367) q[12];
rz(-1.5780799639203424) q[12];
ry(1.5708125753182691) q[13];
rz(2.5387229519191377) q[13];
ry(-3.141592604642369) q[14];
rz(2.7556572252646103) q[14];
ry(3.1415924523705567) q[15];
rz(0.7769084069631605) q[15];
ry(-0.36930151804049444) q[16];
rz(2.902020189697315) q[16];
ry(1.3283748502088717) q[17];
rz(0.028847917930582948) q[17];
ry(-0.027301553625592636) q[18];
rz(2.458191048290192) q[18];
ry(3.135076804513742) q[19];
rz(-1.6287039681789124) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.532847151125988) q[0];
rz(1.2454970434946047) q[0];
ry(1.2401999454300925) q[1];
rz(2.768146136171031) q[1];
ry(1.632346829658613) q[2];
rz(-3.105812218823) q[2];
ry(-1.5448954908624661) q[3];
rz(1.3913279309202409) q[3];
ry(-1.5636211589677407) q[4];
rz(0.058002968210209005) q[4];
ry(1.5704555132953646) q[5];
rz(1.2736962071807163) q[5];
ry(1.5666678707275503) q[6];
rz(1.5492816764594037) q[6];
ry(1.5668541696139728) q[7];
rz(-0.009568380270675194) q[7];
ry(1.577518622748297) q[8];
rz(-0.03874252081410124) q[8];
ry(1.5540649365080739) q[9];
rz(-1.1569481013028133) q[9];
ry(-2.449289985704115) q[10];
rz(-0.9971301173511122) q[10];
ry(0.8088498947744425) q[11];
rz(-1.346848978930697) q[11];
ry(1.575439059635762) q[12];
rz(3.0977170445417084) q[12];
ry(-0.012528812560290469) q[13];
rz(0.6075009226207877) q[13];
ry(1.3283637055216959e-05) q[14];
rz(-2.88389262151596) q[14];
ry(3.1415671184295295) q[15];
rz(0.277546975861724) q[15];
ry(0.07925586640387357) q[16];
rz(-2.7775240739946723) q[16];
ry(1.229677627171529) q[17];
rz(-2.8034770087436627) q[17];
ry(-1.5186744164731312) q[18];
rz(0.6072300480087315) q[18];
ry(1.5738510426311059) q[19];
rz(-3.0744080957734763) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.1322549897009777) q[0];
rz(1.5778443042290178) q[0];
ry(-2.5585181081147796) q[1];
rz(0.6753053595740619) q[1];
ry(-0.045285294707033685) q[2];
rz(2.4871354371478973) q[2];
ry(-0.0028799972555599847) q[3];
rz(-1.8803771230635427) q[3];
ry(-1.5911433749530595) q[4];
rz(1.57075210847699) q[4];
ry(0.03439428446387018) q[5];
rz(0.9715438535449181) q[5];
ry(0.0028540716424005907) q[6];
rz(1.8485246882608406) q[6];
ry(3.13866064413504) q[7];
rz(1.2956487999481228) q[7];
ry(0.03352956781572303) q[8];
rz(1.9214319499263341) q[8];
ry(-3.05519599820032) q[9];
rz(0.5319129960953123) q[9];
ry(-0.007708120268977048) q[10];
rz(-1.8548733546771103) q[10];
ry(0.010945387404404094) q[11];
rz(0.18197262775985476) q[11];
ry(-3.0859388617750776) q[12];
rz(-1.6035032043255395) q[12];
ry(-0.9146924498454068) q[13];
rz(0.16681064530103829) q[13];
ry(-1.5221598668045022) q[14];
rz(-1.574904705849859) q[14];
ry(1.5586255951010548) q[15];
rz(-2.2156871968770364) q[15];
ry(-1.5661910889187052) q[16];
rz(-1.5698520877756401) q[16];
ry(0.06789049363412614) q[17];
rz(1.3591576711529756) q[17];
ry(-3.0204736188337713) q[18];
rz(0.777164317129217) q[18];
ry(-1.4706374330699568) q[19];
rz(0.1680433886433974) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.575350625699234) q[0];
rz(2.0769603105189844) q[0];
ry(-3.133127987379661) q[1];
rz(0.10311626734240774) q[1];
ry(3.1398381351584472) q[2];
rz(-0.5897111886503588) q[2];
ry(-0.0012749524629123957) q[3];
rz(-2.5721863655738946) q[3];
ry(-1.5489271077701074) q[4];
rz(0.5686432641304586) q[4];
ry(-0.02854578886068193) q[5];
rz(-0.6760164277948093) q[5];
ry(1.5922713306666605) q[6];
rz(0.21039707637146066) q[6];
ry(1.4778509199485104) q[7];
rz(0.13445700832054183) q[7];
ry(0.023330066697514678) q[8];
rz(-0.34726948365860455) q[8];
ry(0.48084706502299235) q[9];
rz(2.751463644920309) q[9];
ry(-1.0707149001179446) q[10];
rz(-1.4740442367470834) q[10];
ry(-1.1953827495026463) q[11];
rz(-1.4314517000739533) q[11];
ry(3.130800426162579) q[12];
rz(-0.4835847120932183) q[12];
ry(-3.0768790767491585) q[13];
rz(-2.974490101114691) q[13];
ry(3.1414693181561617) q[14];
rz(3.136108336534461) q[14];
ry(-3.1413212097944267) q[15];
rz(-0.6426452099258123) q[15];
ry(1.5715612439110638) q[16];
rz(1.7924943515801495) q[16];
ry(-0.00043986228851355946) q[17];
rz(1.0045129545872884) q[17];
ry(1.6097569920471886) q[18];
rz(0.05038411456707676) q[18];
ry(-1.5703749537439116) q[19];
rz(-2.645704720644435) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.0589463089801645) q[0];
rz(-0.4543962850724848) q[0];
ry(0.012496975231993446) q[1];
rz(0.6808469586700935) q[1];
ry(1.1943483187622945) q[2];
rz(-0.7377654320000842) q[2];
ry(-0.3914965753524271) q[3];
rz(-3.120655748832239) q[3];
ry(1.5401253683553162) q[4];
rz(1.6071691258951661) q[4];
ry(1.5920505110388854) q[5];
rz(3.1003503296881854) q[5];
ry(2.281447981585943) q[6];
rz(0.18002360001714646) q[6];
ry(0.8499749843985325) q[7];
rz(-0.1767661486526393) q[7];
ry(3.1401743829462263) q[8];
rz(-1.305085512698712) q[8];
ry(-0.0028112334487397916) q[9];
rz(-2.7908323697560475) q[9];
ry(-0.042946563207096895) q[10];
rz(-0.3602446663415164) q[10];
ry(3.1293764775322837) q[11];
rz(-0.1745449447883359) q[11];
ry(-3.125766818603106) q[12];
rz(1.081562069147826) q[12];
ry(-1.4980539256980352) q[13];
rz(-1.5762071089434762) q[13];
ry(2.9018735588634623) q[14];
rz(2.3693406912185773) q[14];
ry(1.3331493067740245) q[15];
rz(-2.3701618156865725) q[15];
ry(-3.135356265287402) q[16];
rz(-2.261530798572229) q[16];
ry(-1.5764173265637524) q[17];
rz(-2.7620742716062807) q[17];
ry(1.2934848488895856) q[18];
rz(-1.551902593866854) q[18];
ry(0.7110214904248008) q[19];
rz(1.9152544250786403) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.6283267330216993) q[0];
rz(-1.279150911448954) q[0];
ry(2.055423738565204) q[1];
rz(0.22549354516991413) q[1];
ry(0.11747414508181998) q[2];
rz(-2.5601421756874037) q[2];
ry(1.3611347763103456) q[3];
rz(-2.874090621434941) q[3];
ry(-0.9940361052923423) q[4];
rz(2.9409634759238172) q[4];
ry(-1.5867094318352704) q[5];
rz(-1.58419102546699) q[5];
ry(-1.5693036629632164) q[6];
rz(-1.570329462521033) q[6];
ry(-1.5693601855843644) q[7];
rz(1.5719250190377874) q[7];
ry(0.003342987740230008) q[8];
rz(1.3059898341190674) q[8];
ry(0.8423366650184442) q[9];
rz(-2.4186965776001483) q[9];
ry(-0.9464847202138396) q[10];
rz(-1.6536793385376616) q[10];
ry(-2.6157607517102788) q[11];
rz(0.6589109275150591) q[11];
ry(1.5752953969418422) q[12];
rz(-3.1327850227611425) q[12];
ry(-1.5745097361381113) q[13];
rz(-3.140073781094375) q[13];
ry(-1.5199883965470806) q[14];
rz(1.0608201518289067) q[14];
ry(-1.5735473325807428) q[15];
rz(-0.883813476125726) q[15];
ry(-3.1374509381169013) q[16];
rz(2.238957534426757) q[16];
ry(3.1366777894661593) q[17];
rz(1.9619031927545565) q[17];
ry(0.0842697950389515) q[18];
rz(-0.43385170169100906) q[18];
ry(3.046194374184423) q[19];
rz(0.9282590527462521) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.137937351919282) q[0];
rz(-3.1194680841919378) q[0];
ry(-0.050298428396788) q[1];
rz(-1.5094092668656829) q[1];
ry(0.09331853310108419) q[2];
rz(-1.4925333820091446) q[2];
ry(3.1232581273411646) q[3];
rz(1.9144086773241684) q[3];
ry(0.0013010834844955355) q[4];
rz(-1.2976592581696451) q[4];
ry(-3.1020459073062305) q[5];
rz(3.131347993393473) q[5];
ry(1.533967951167595) q[6];
rz(-1.6009153502251319) q[6];
ry(-1.5406957606368605) q[7];
rz(-1.570262104139699) q[7];
ry(-0.8214018432330483) q[8];
rz(2.592780294619177) q[8];
ry(-1.6332986383992438) q[9];
rz(-0.1646264202903085) q[9];
ry(0.00556094196806356) q[10];
rz(2.7434099363075592) q[10];
ry(3.089219349954368) q[11];
rz(-0.37019237662148974) q[11];
ry(-1.5690914590763885) q[12];
rz(-2.8086187447911923) q[12];
ry(-1.2918916460881036) q[13];
rz(3.134457350764993) q[13];
ry(3.134436590050871) q[14];
rz(2.654854243330758) q[14];
ry(3.0746983838063477) q[15];
rz(0.7091454554479989) q[15];
ry(-2.620498176014488) q[16];
rz(3.051067255645178) q[16];
ry(2.612986581778375) q[17];
rz(2.0809729884938424) q[17];
ry(0.5924420488030497) q[18];
rz(-2.1809815727051887) q[18];
ry(0.5479935384791387) q[19];
rz(0.6723137170168711) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.6001549668010466) q[0];
rz(-0.37886098422680436) q[0];
ry(1.5452165383538867) q[1];
rz(-2.9596024621513104) q[1];
ry(-1.575857444983729) q[2];
rz(2.60450951662875) q[2];
ry(1.5582546666152943) q[3];
rz(1.78184861431084) q[3];
ry(-2.615362103588943) q[4];
rz(1.0793537555438344) q[4];
ry(-2.6168556825974503) q[5];
rz(-1.7529042993084358) q[5];
ry(1.6145466683383045) q[6];
rz(-2.9279436123246168) q[6];
ry(-1.5792273042316207) q[7];
rz(2.9488375945184755) q[7];
ry(0.0032076232973057887) q[8];
rz(1.8104766138672745) q[8];
ry(3.1158584381051635) q[9];
rz(-3.0377911388871057) q[9];
ry(3.134383232234823) q[10];
rz(-0.8265058935010691) q[10];
ry(-0.011878704420143818) q[11];
rz(-2.8108767599449718) q[11];
ry(-1.570692158048532) q[12];
rz(1.5534019407448183) q[12];
ry(1.5897385660892827) q[13];
rz(3.1412386068696234) q[13];
ry(1.5696604801342728) q[14];
rz(3.1182537003995083) q[14];
ry(1.5719187600246451) q[15];
rz(-1.5498916083745558) q[15];
ry(0.0025086778435184248) q[16];
rz(-0.005311317141955207) q[16];
ry(3.1376157321907447) q[17];
rz(2.0726412978788487) q[17];
ry(-3.134145324139235) q[18];
rz(0.05402333143359315) q[18];
ry(1.4384133518566082) q[19];
rz(1.584711773927614) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.72679354516859) q[0];
rz(1.8629839932010208) q[0];
ry(-1.7156640449525231) q[1];
rz(1.851952102396462) q[1];
ry(-2.7995470948274495) q[2];
rz(-0.154482222421553) q[2];
ry(0.41402515265038) q[3];
rz(-0.08963256435639169) q[3];
ry(-2.75887264550074) q[4];
rz(-0.18889168178190816) q[4];
ry(1.24686396046667) q[5];
rz(-2.811285093144142) q[5];
ry(-2.7548485219020837) q[6];
rz(-0.26575837884429393) q[6];
ry(0.385419564471956) q[7];
rz(-0.2835056357763216) q[7];
ry(1.2297247337713013) q[8];
rz(-2.7652511717122885) q[8];
ry(1.9045919217079197) q[9];
rz(0.42741238468742115) q[9];
ry(1.5916747083269298) q[10];
rz(1.1312190070356734) q[10];
ry(1.5908267965801093) q[11];
rz(1.1303109159176083) q[11];
ry(-1.575373062670724) q[12];
rz(2.702988482792383) q[12];
ry(-1.5781382476894652) q[13];
rz(-2.003873786811724) q[13];
ry(-3.1381123961451007) q[14];
rz(-0.4513440028070094) q[14];
ry(-0.015931430148259373) q[15];
rz(-2.019465436404213) q[15];
ry(-0.03051374378921957) q[16];
rz(1.2544740465658666) q[16];
ry(-1.877384792104892) q[17];
rz(-1.9938868776322725) q[17];
ry(0.01753971453869063) q[18];
rz(1.5568440791401699) q[18];
ry(-1.5774260019881963) q[19];
rz(0.6658857980520878) q[19];