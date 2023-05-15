OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.15743211458734319) q[0];
rz(1.5476203268976776) q[0];
ry(-0.7002571590575881) q[1];
rz(3.046295219981517) q[1];
ry(-2.521614685607195) q[2];
rz(-0.2933372734536112) q[2];
ry(1.2008787520277262) q[3];
rz(-3.111118146273113) q[3];
ry(1.287606100335104) q[4];
rz(2.793093997222674) q[4];
ry(-0.8382582065792147) q[5];
rz(2.811829446097794) q[5];
ry(2.636439747681598) q[6];
rz(2.977709932537901) q[6];
ry(-1.0623973909542377) q[7];
rz(2.8668172808376022) q[7];
ry(1.8401951195047292) q[8];
rz(-0.30013813794609007) q[8];
ry(-1.1469228141103733) q[9];
rz(-0.07303931943811932) q[9];
ry(-2.47109910523396) q[10];
rz(-0.07873088553828199) q[10];
ry(2.6711338907963746) q[11];
rz(-0.03998112290635854) q[11];
ry(-2.0935702332890154) q[12];
rz(3.032222423623349) q[12];
ry(-1.4787567020186074) q[13];
rz(-2.809815889819064) q[13];
ry(0.804632019957891) q[14];
rz(0.056866544402164904) q[14];
ry(-2.548492187714052) q[15];
rz(0.0731364393945695) q[15];
ry(-0.9733815066723409) q[16];
rz(-0.06343405551915648) q[16];
ry(0.9821791214805825) q[17];
rz(-2.6862157933908892) q[17];
ry(1.9131006612537726) q[18];
rz(-0.9733823516159833) q[18];
ry(2.850598140226671) q[19];
rz(-1.5551714214889791) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.9456779103084678) q[0];
rz(0.9216201327369857) q[0];
ry(-0.9517808739670235) q[1];
rz(0.038811971967581016) q[1];
ry(2.691361779589003) q[2];
rz(0.3175806156253696) q[2];
ry(0.5236072450752328) q[3];
rz(-2.5345824909911046) q[3];
ry(2.792731937266026) q[4];
rz(-0.15506184803771195) q[4];
ry(0.39933575752258216) q[5];
rz(-0.2885217903754709) q[5];
ry(0.5131177671568077) q[6];
rz(-0.5419043113008336) q[6];
ry(-2.6679782137816965) q[7];
rz(-3.0976192602964225) q[7];
ry(2.754665840622412) q[8];
rz(-3.070786239065593) q[8];
ry(2.6622382252539225) q[9];
rz(0.3817527056854487) q[9];
ry(2.791547878590778) q[10];
rz(-2.6644314279969645) q[10];
ry(2.789764784698241) q[11];
rz(-0.7678399920381501) q[11];
ry(2.4461138249180263) q[12];
rz(0.3081075568822147) q[12];
ry(-2.7327263061957936) q[13];
rz(-3.0443739281048137) q[13];
ry(0.4668768086495469) q[14];
rz(-2.8439386435582192) q[14];
ry(-2.5239619277745886) q[15];
rz(2.33872285918792) q[15];
ry(-0.1656323495295542) q[16];
rz(1.3786513159655227) q[16];
ry(2.8076559350944037) q[17];
rz(-1.7668112428261393) q[17];
ry(1.9865849020074058) q[18];
rz(1.9900942462298266) q[18];
ry(-1.427024467536394) q[19];
rz(1.5168034779334933) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.4085732958145458) q[0];
rz(1.570327197052742) q[0];
ry(-0.6504009085168088) q[1];
rz(2.9528940292184283) q[1];
ry(1.9513325832029098) q[2];
rz(1.955318142580275) q[2];
ry(-1.54701860688091) q[3];
rz(2.569220509806473) q[3];
ry(1.0351994633494943) q[4];
rz(-0.38191203347871244) q[4];
ry(-1.95695946722583) q[5];
rz(-2.8804917488854382) q[5];
ry(0.6919538121284461) q[6];
rz(0.001280081240834896) q[6];
ry(2.1420045897934568) q[7];
rz(3.052107112945181) q[7];
ry(-1.1575227541216195) q[8];
rz(0.08632452933102676) q[8];
ry(0.9742039746451747) q[9];
rz(0.10751881192215894) q[9];
ry(-1.9634547840150276) q[10];
rz(-0.3702595961357886) q[10];
ry(0.8275057324305666) q[11];
rz(-0.37383098656263947) q[11];
ry(0.7419782125273837) q[12];
rz(-2.9902575781032312) q[12];
ry(-1.8671729405702227) q[13];
rz(-3.095988687079133) q[13];
ry(0.7092816026786171) q[14];
rz(-2.7576026801391533) q[14];
ry(0.9848725095223818) q[15];
rz(2.172633693911783) q[15];
ry(1.930455432730707) q[16];
rz(1.4125795697903678) q[16];
ry(-2.3026149325779017) q[17];
rz(-0.3742946627437397) q[17];
ry(2.65840195600623) q[18];
rz(2.777585628315827) q[18];
ry(1.4703360536961103) q[19];
rz(-1.9804625057369243) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.1276798834146824) q[0];
rz(2.533622395520276) q[0];
ry(1.2233154888462974) q[1];
rz(1.651590325384498) q[1];
ry(-3.0049649048728058) q[2];
rz(0.5557143535707403) q[2];
ry(2.941705164565767) q[3];
rz(1.525436682864749) q[3];
ry(0.46083033422441394) q[4];
rz(-1.4877066318414467) q[4];
ry(2.8335074037945938) q[5];
rz(-1.9592993774661513) q[5];
ry(2.778475050957215) q[6];
rz(1.9323784629406082) q[6];
ry(-0.709408158294222) q[7];
rz(1.441020027783984) q[7];
ry(0.6049910573254422) q[8];
rz(1.656368699553731) q[8];
ry(2.7946060411463303) q[9];
rz(-1.8729179462668162) q[9];
ry(2.7886544335856924) q[10];
rz(-1.360345261559221) q[10];
ry(-0.3279555818450991) q[11];
rz(1.4087165744627708) q[11];
ry(-2.6354323729786726) q[12];
rz(1.38888018269457) q[12];
ry(1.9919654214135276) q[13];
rz(1.422696612848705) q[13];
ry(0.44821319353444533) q[14];
rz(-1.3607369785490384) q[14];
ry(2.7362620661144805) q[15];
rz(-1.6871081807682353) q[15];
ry(3.0565141523338624) q[16];
rz(1.8251772970503235) q[16];
ry(-2.97399375477737) q[17];
rz(1.3421402097602124) q[17];
ry(2.54926009183506) q[18];
rz(-1.675291555178355) q[18];
ry(1.5311678943043097) q[19];
rz(1.4672960065265395) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.06546588848415733) q[0];
rz(-1.803152195758985) q[0];
ry(-1.6185785830445403) q[1];
rz(-0.9676804801146) q[1];
ry(2.0324905962905966) q[2];
rz(-0.8607675104536293) q[2];
ry(0.5543341498181933) q[3];
rz(-0.8514873431244441) q[3];
ry(-1.4034936350285516) q[4];
rz(-0.5069990920325109) q[4];
ry(-2.0676517191965456) q[5];
rz(-1.809509604752435) q[5];
ry(2.1540005677448497) q[6];
rz(-0.7309496862366942) q[6];
ry(-2.0503966574685615) q[7];
rz(-1.3159351421699963) q[7];
ry(-2.2169397406521654) q[8];
rz(2.218513536947012) q[8];
ry(2.328668287685293) q[9];
rz(2.2199968656237834) q[9];
ry(-1.852101944921763) q[10];
rz(2.905466357028929) q[10];
ry(2.0534672395837026) q[11];
rz(1.5786994485596293) q[11];
ry(-0.7911190789697268) q[12];
rz(-1.172417048378433) q[12];
ry(1.9925940583081534) q[13];
rz(1.943426509279801) q[13];
ry(-2.2440442457010916) q[14];
rz(2.365030425015343) q[14];
ry(-1.1934026754385076) q[15];
rz(1.6407060176212385) q[15];
ry(-1.5022865686045614) q[16];
rz(0.6731330117101271) q[16];
ry(-1.895880164097246) q[17];
rz(2.7365591721767792) q[17];
ry(-2.201903001462415) q[18];
rz(2.3003762893089537) q[18];
ry(2.679111405678525) q[19];
rz(2.0959048974516903) q[19];