OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5707732375063321) q[0];
rz(3.1415521879914086) q[0];
ry(1.5707894973989518) q[1];
rz(0.0011313814646109921) q[1];
ry(0.7924357744794417) q[2];
rz(-2.7116949878274257) q[2];
ry(-1.7635787198098738) q[3];
rz(-2.136916934451132) q[3];
ry(1.689526167087025) q[4];
rz(-1.2729274246948699) q[4];
ry(-2.0135218213951425) q[5];
rz(2.318959660738325) q[5];
ry(1.5613303033432029) q[6];
rz(0.25916251078766894) q[6];
ry(-1.572043659470929) q[7];
rz(-0.0056209243841368415) q[7];
ry(0.7526950997139892) q[8];
rz(-2.7487489400448344) q[8];
ry(2.109748259152677) q[9];
rz(0.3353847696206396) q[9];
ry(-2.848884244382941) q[10];
rz(0.09297716458637452) q[10];
ry(-0.2259398976428467) q[11];
rz(-0.7207347745451094) q[11];
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
ry(2.167360926005215) q[0];
rz(3.0192751873641583) q[0];
ry(0.8353431140667599) q[1];
rz(-1.3395103867505067) q[1];
ry(2.502101551095771) q[2];
rz(-1.0814586556024697) q[2];
ry(-0.8902764808766381) q[3];
rz(-1.364988365653689) q[3];
ry(1.0112433939661445) q[4];
rz(2.624407869290747) q[4];
ry(1.3514364338480818) q[5];
rz(-1.9252125240928155) q[5];
ry(-0.00823044919297962) q[6];
rz(1.3113633465767705) q[6];
ry(-0.9552680827529079) q[7];
rz(-1.569583956647676) q[7];
ry(2.4405387442293316) q[8];
rz(-1.7081072085623576) q[8];
ry(-2.974009419885071) q[9];
rz(0.894662720187704) q[9];
ry(-2.6543412309328027) q[10];
rz(0.6674243800617603) q[10];
ry(-0.29963900258938503) q[11];
rz(-0.18840868603204738) q[11];
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
ry(0.00014990139867698815) q[0];
rz(0.12232688275572554) q[0];
ry(-3.138339093006832) q[1];
rz(-2.4850195678349034) q[1];
ry(-0.9775542834239365) q[2];
rz(2.0197576242244577) q[2];
ry(-0.910473824358097) q[3];
rz(-1.7565772370952173) q[3];
ry(1.5525950046947008) q[4];
rz(-1.0147552225768228) q[4];
ry(-2.7966559950021135) q[5];
rz(-2.0835931749601153) q[5];
ry(-2.137599718838061) q[6];
rz(-1.2667693130411735) q[6];
ry(1.438917110213826) q[7];
rz(-0.2139006225589064) q[7];
ry(-2.094175544627545) q[8];
rz(2.743585073776159) q[8];
ry(0.5611205057850279) q[9];
rz(1.2379844119431478) q[9];
ry(-0.7973894941975957) q[10];
rz(-0.09767216236135479) q[10];
ry(2.2546211922947257) q[11];
rz(-2.5015597983584232) q[11];
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
ry(-1.4217467751202915) q[0];
rz(-1.570690257655559) q[0];
ry(3.1415283772543843) q[1];
rz(-2.7162424492519457) q[1];
ry(-1.534677426937277) q[2];
rz(-3.002743175860373) q[2];
ry(0.753289771515322) q[3];
rz(-0.35645718024062306) q[3];
ry(-0.46805043688027226) q[4];
rz(-2.1514180818832087) q[4];
ry(1.6536486206209569) q[5];
rz(-1.135542892136486) q[5];
ry(3.130571624050932) q[6];
rz(1.8762223773176119) q[6];
ry(-0.00015579588553826795) q[7];
rz(2.208613126498179) q[7];
ry(-1.6752992549904162) q[8];
rz(1.9125971649139473) q[8];
ry(-1.1617983722187442) q[9];
rz(0.17590018948716502) q[9];
ry(-0.8274570287770082) q[10];
rz(-1.9361016511249025) q[10];
ry(-1.3945214640452122) q[11];
rz(-0.367391817963596) q[11];
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
ry(2.389631155737018) q[0];
rz(-0.7736405589548118) q[0];
ry(1.3533755448773697) q[1];
rz(-0.5415831046163877) q[1];
ry(3.090350788096079) q[2];
rz(-0.7895437044239824) q[2];
ry(1.4137090005777493) q[3];
rz(2.021567161614751) q[3];
ry(-0.8847155652687385) q[4];
rz(2.581999645359456) q[4];
ry(-1.6691379841852436) q[5];
rz(2.538195906985477) q[5];
ry(2.007581918755367) q[6];
rz(-0.03544496901348425) q[6];
ry(-3.1376108829324134) q[7];
rz(-2.7165031036571525) q[7];
ry(1.8381766549253993) q[8];
rz(3.1245661551531887) q[8];
ry(-1.6422302032767253) q[9];
rz(2.500351326413948) q[9];
ry(2.7104596062977535) q[10];
rz(2.2768253458260514) q[10];
ry(0.34935367404748696) q[11];
rz(-1.072929037805787) q[11];
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
ry(-3.141531533575267) q[0];
rz(2.367941331329361) q[0];
ry(0.0007725929031163759) q[1];
rz(0.592594839435554) q[1];
ry(-0.7738964639387492) q[2];
rz(-0.3515208792361403) q[2];
ry(-1.324234382685539) q[3];
rz(3.078539433472647) q[3];
ry(2.2143105988007243) q[4];
rz(3.027088066380398) q[4];
ry(-1.5540390420337844) q[5];
rz(3.0020777850298517) q[5];
ry(0.003432261282993723) q[6];
rz(3.0085756128361663) q[6];
ry(-0.9767781969909667) q[7];
rz(1.5803547281040542) q[7];
ry(2.7353707247794734) q[8];
rz(2.3404361483504577) q[8];
ry(2.299018666734531) q[9];
rz(1.913498859336526) q[9];
ry(-1.214382089515827) q[10];
rz(-1.6357640095106385) q[10];
ry(2.15762802760468) q[11];
rz(2.084511881339755) q[11];
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
ry(-1.5026394888565724) q[0];
rz(1.5707646653595049) q[0];
ry(-3.096095140115779) q[1];
rz(-1.8721587955382961) q[1];
ry(-1.6910682663598968) q[2];
rz(1.4598137815404435) q[2];
ry(-1.9291637151474506) q[3];
rz(-0.5466480275726697) q[3];
ry(2.527556266736381) q[4];
rz(2.2409995276772126) q[4];
ry(1.3856273248015274) q[5];
rz(-0.5017328891946233) q[5];
ry(-0.0055529313540043646) q[6];
rz(-1.6874048645084745) q[6];
ry(0.4104072374301211) q[7];
rz(0.21462695384330513) q[7];
ry(2.3311000128416413) q[8];
rz(-2.398359457900595) q[8];
ry(-1.1568799248115145) q[9];
rz(-2.672207988411804) q[9];
ry(1.0202399649586749) q[10];
rz(-0.987997942203544) q[10];
ry(2.5270873041025537) q[11];
rz(-0.8256712272381996) q[11];
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
ry(-0.767925394334962) q[0];
rz(-1.7683081621944723) q[0];
ry(0.0001816264290203182) q[1];
rz(-1.221575256439543) q[1];
ry(-0.2546486666919394) q[2];
rz(-2.779550323274074) q[2];
ry(2.1032278168394765) q[3];
rz(2.4075846296742105) q[3];
ry(0.614935372297853) q[4];
rz(-1.5923289660531823) q[4];
ry(0.24670085429567212) q[5];
rz(0.6778976827844688) q[5];
ry(-0.01108325781609281) q[6];
rz(-1.284104705457652) q[6];
ry(3.137843595901481) q[7];
rz(0.2511146271302721) q[7];
ry(-0.4161278451107018) q[8];
rz(1.0052010398063584) q[8];
ry(2.0823014933602977) q[9];
rz(-2.4417385060488677) q[9];
ry(-1.4528086827119802) q[10];
rz(1.8686482041182133) q[10];
ry(-2.6907398775987517) q[11];
rz(-1.8149739835606216) q[11];
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
ry(-3.1406819060156272) q[0];
rz(-1.7683903995721681) q[0];
ry(-0.9752395318988496) q[1];
rz(1.8534378306347055) q[1];
ry(2.430058604048199) q[2];
rz(-2.6004887126594367) q[2];
ry(-1.3146388881775284) q[3];
rz(-0.030025090667217036) q[3];
ry(0.49600226393671987) q[4];
rz(-0.4139472818404978) q[4];
ry(1.3689764769177812) q[5];
rz(2.5205714342537475) q[5];
ry(-0.7724699743893826) q[6];
rz(1.2437286697923164) q[6];
ry(-0.15903751022805945) q[7];
rz(2.8308643663890227) q[7];
ry(-1.6915908715049266) q[8];
rz(-1.7369601486970068) q[8];
ry(2.3362464658318984) q[9];
rz(-0.7863106839650968) q[9];
ry(-1.955552855322182) q[10];
rz(2.234711103850408) q[10];
ry(-2.727468463279128) q[11];
rz(2.4648938504216984) q[11];
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
ry(-2.2003250763696824) q[0];
rz(-1.5706132774453776) q[0];
ry(3.133004582019786) q[1];
rz(-1.2907633643823617) q[1];
ry(0.9312738986948709) q[2];
rz(-1.2113074404557753) q[2];
ry(2.706175130688294) q[3];
rz(-0.2623988741210219) q[3];
ry(1.7868634397371261) q[4];
rz(-3.1013359836623953) q[4];
ry(-0.4303722083560346) q[5];
rz(-2.0450023699553452) q[5];
ry(-3.1352404817926542) q[6];
rz(1.2498817351397227) q[6];
ry(-3.1400468405406223) q[7];
rz(2.8595513340374805) q[7];
ry(2.330748393549516) q[8];
rz(1.131373852793413) q[8];
ry(1.797310074725512) q[9];
rz(3.1297010402940946) q[9];
ry(2.99533931230349) q[10];
rz(0.7824037921661353) q[10];
ry(-2.7569509217654415) q[11];
rz(-0.5146408332391301) q[11];
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
ry(-0.09976237416986006) q[0];
rz(1.1603407000237673) q[0];
ry(2.275514854408258) q[1];
rz(-1.5720369474464004) q[1];
ry(2.5849751041947884) q[2];
rz(-1.4628404637360202) q[2];
ry(1.5059331378622858) q[3];
rz(1.2052724686385563) q[3];
ry(1.060667146765837) q[4];
rz(0.35404191700903104) q[4];
ry(1.5308791189962108) q[5];
rz(-1.969661737330405) q[5];
ry(0.2568649863334871) q[6];
rz(-1.5755989335383136) q[6];
ry(-1.9938968841919316) q[7];
rz(-2.514293384845385) q[7];
ry(1.6958764940702968) q[8];
rz(3.1111512370825563) q[8];
ry(1.2253740333460437) q[9];
rz(-2.515361702751058) q[9];
ry(2.7792331943419097) q[10];
rz(-1.1129688003095977) q[10];
ry(-2.296388900130161) q[11];
rz(-2.9794968443980103) q[11];
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
ry(3.1413555250455563) q[0];
rz(-1.9875890645808791) q[0];
ry(1.1247265110796842) q[1];
rz(1.5743167602767638) q[1];
ry(-0.784140264245492) q[2];
rz(0.047608250141113286) q[2];
ry(-1.0668888312903588) q[3];
rz(0.2151745974553783) q[3];
ry(-1.7007589463661008) q[4];
rz(-2.3933588302410715) q[4];
ry(2.435094612100872) q[5];
rz(1.4438228729318718) q[5];
ry(0.48921872510453945) q[6];
rz(1.5757285566311863) q[6];
ry(-3.1390600105391346) q[7];
rz(-2.130096658730075) q[7];
ry(1.9934931616404628) q[8];
rz(2.529497891306835) q[8];
ry(-2.5065324066476866) q[9];
rz(1.696132891553265) q[9];
ry(2.8146110138259814) q[10];
rz(0.23832827198536272) q[10];
ry(-1.6679826397342288) q[11];
rz(-1.8795723273119724) q[11];
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
ry(3.1210207310708764) q[0];
rz(-0.1113224930348311) q[0];
ry(1.2645378383325259) q[1];
rz(1.5703980940236617) q[1];
ry(-2.625774783689204) q[2];
rz(0.3082202315113806) q[2];
ry(-2.661217125049065) q[3];
rz(2.37814921453282) q[3];
ry(-0.7372631671801748) q[4];
rz(-1.2407292596069337) q[4];
ry(-1.0053013102101565) q[5];
rz(1.185156778684384) q[5];
ry(-1.8306944772767246) q[6];
rz(-1.568794515804072) q[6];
ry(0.00700061232961513) q[7];
rz(-1.9557579925349282) q[7];
ry(1.2830778342133438) q[8];
rz(2.809472356009327) q[8];
ry(0.525416695400075) q[9];
rz(0.5193029853590128) q[9];
ry(-2.2979092318388985) q[10];
rz(-1.6711116600013687) q[10];
ry(1.5991917013609909) q[11];
rz(-1.8736401570927945) q[11];
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
ry(4.870795800648864e-05) q[0];
rz(0.10472379164470944) q[0];
ry(-2.1200643838253583) q[1];
rz(-1.8047726200368102) q[1];
ry(2.291533476773031) q[2];
rz(-1.1442944180611239) q[2];
ry(1.5756090423495952) q[3];
rz(2.482808896558573) q[3];
ry(-2.0219581371043756) q[4];
rz(-1.4864959948865222) q[4];
ry(-2.3882377947771545) q[5];
rz(-2.488314085725475) q[5];
ry(-2.6757248806855167) q[6];
rz(1.9894439753547566) q[6];
ry(-0.9345495557243901) q[7];
rz(0.035438334764275936) q[7];
ry(1.0566665208794452) q[8];
rz(2.4553370747887784) q[8];
ry(-2.212750026806278) q[9];
rz(-0.045637614220330525) q[9];
ry(0.13070980358066017) q[10];
rz(-0.4453568578609843) q[10];
ry(-0.9834231565250767) q[11];
rz(-2.197148365219179) q[11];
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
ry(-0.417999135980785) q[0];
rz(-3.109956110718992) q[0];
ry(-3.1274476278102403) q[1];
rz(-0.3398257171865638) q[1];
ry(0.5267471836576877) q[2];
rz(0.060464634611872015) q[2];
ry(2.8572519730459187) q[3];
rz(-1.054402156967286) q[3];
ry(0.7166351456852276) q[4];
rz(2.527343279111554) q[4];
ry(-2.2077674685766118) q[5];
rz(-1.4661685646890765) q[5];
ry(3.1334044836651715) q[6];
rz(1.9882768586925659) q[6];
ry(3.1387671170701057) q[7];
rz(2.9443592269477405) q[7];
ry(2.5435880560529713) q[8];
rz(2.2101067022372707) q[8];
ry(1.8838381660989925) q[9];
rz(1.6486566100319955) q[9];
ry(2.6608866062269603) q[10];
rz(3.0476198794570335) q[10];
ry(0.38445424437460485) q[11];
rz(-2.952812449795808) q[11];
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
ry(3.1414211295927554) q[0];
rz(0.03142814664491844) q[0];
ry(3.1382124762718933) q[1];
rz(2.8956632035418304) q[1];
ry(0.6549508085930987) q[2];
rz(-1.9864826211812643) q[2];
ry(0.28669933151349714) q[3];
rz(2.4945444917386386) q[3];
ry(-1.7758253411896971) q[4];
rz(-0.5965651250142425) q[4];
ry(0.7792593219416064) q[5];
rz(0.9931098184517853) q[5];
ry(2.59155820840393) q[6];
rz(-1.5767203396564466) q[6];
ry(0.005159052533917574) q[7];
rz(1.7902330055572047) q[7];
ry(-1.0617540767009068) q[8];
rz(-1.6628067168254903) q[8];
ry(2.0296001800345893) q[9];
rz(-2.016967024785573) q[9];
ry(1.8849537620401913) q[10];
rz(-0.6309107639087728) q[10];
ry(0.8485443851372461) q[11];
rz(-0.23062327125721183) q[11];
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
ry(2.4575844265254143) q[0];
rz(1.5705897545614222) q[0];
ry(-3.1263853887188464) q[1];
rz(1.425050014465083) q[1];
ry(-1.281067165859958) q[2];
rz(-2.1087902441719204) q[2];
ry(-2.9895575280769933) q[3];
rz(2.574244552777718) q[3];
ry(1.3225587547787168) q[4];
rz(-0.5064904891958188) q[4];
ry(-0.27453980119837684) q[5];
rz(-2.680203815963256) q[5];
ry(-0.8130087843165225) q[6];
rz(1.0360354882538647) q[6];
ry(2.8994383695386046) q[7];
rz(-2.4094099424539377) q[7];
ry(1.8218708662616172) q[8];
rz(3.082492027813337) q[8];
ry(-2.9935234939209234) q[9];
rz(-2.518089582740969) q[9];
ry(-2.0046452679822235) q[10];
rz(-0.0033583982171991877) q[10];
ry(-1.6084766606724852) q[11];
rz(0.377479253333324) q[11];
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
ry(1.5344313903349145) q[0];
rz(1.5708027621980585) q[0];
ry(2.8382787208929234) q[1];
rz(1.7198275810451396) q[1];
ry(-0.9493990067689624) q[2];
rz(2.17740168460384) q[2];
ry(0.6190661916179252) q[3];
rz(-2.980420706487107) q[3];
ry(1.674605403220756) q[4];
rz(-2.5053377647477135) q[4];
ry(-2.7388013991722984) q[5];
rz(-1.9042485094037451) q[5];
ry(-0.002349560215632633) q[6];
rz(-1.0300521696526745) q[6];
ry(-0.0030731816535141125) q[7];
rz(-0.7470595176750806) q[7];
ry(2.4654467666313287) q[8];
rz(1.0429395089502718) q[8];
ry(-1.0085378351140157) q[9];
rz(-2.082282249848067) q[9];
ry(1.7453332124153844) q[10];
rz(-2.6737833826613686) q[10];
ry(3.051042924480365) q[11];
rz(-0.45352033223927896) q[11];
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
ry(-0.911535798882859) q[0];
rz(-3.0104821272271796) q[0];
ry(-0.016902630513614625) q[1];
rz(-1.7295982235926532) q[1];
ry(-1.1855596487074556) q[2];
rz(2.698267256785595) q[2];
ry(0.4530043377359858) q[3];
rz(-1.4937659381321247) q[3];
ry(-0.6013801099812959) q[4];
rz(-1.3427401588232692) q[4];
ry(-0.5462144980183847) q[5];
rz(-0.426379841853823) q[5];
ry(-1.2400743455253682) q[6];
rz(1.5686320908800173) q[6];
ry(1.1258135449623405) q[7];
rz(2.887612975673783) q[7];
ry(2.7930842342504985) q[8];
rz(-3.1028930169300275) q[8];
ry(2.210295584144819) q[9];
rz(-0.4195207924014554) q[9];
ry(-2.1641184517868184) q[10];
rz(2.7744052993006987) q[10];
ry(0.5587412615303912) q[11];
rz(1.824930428149126) q[11];
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
ry(3.1415924214371467) q[0];
rz(2.7901624197825616) q[0];
ry(0.45944638803868454) q[1];
rz(1.5783822022297282) q[1];
ry(0.863710935155626) q[2];
rz(2.755508365712315) q[2];
ry(1.252076731051178) q[3];
rz(1.1578172906171473) q[3];
ry(1.7937948101428214) q[4];
rz(-1.3075255698340396) q[4];
ry(2.0906948718628913) q[5];
rz(-3.110436102273578) q[5];
ry(-2.009195328293472) q[6];
rz(1.5759542700732023) q[6];
ry(0.00267193261530881) q[7];
rz(-2.8816059175057034) q[7];
ry(0.8723806008681114) q[8];
rz(-1.9313262583195407) q[8];
ry(2.2105015700083532) q[9];
rz(2.0376563633615334) q[9];
ry(-3.0271909634692666) q[10];
rz(1.1799373008711498) q[10];
ry(1.7853529269726884) q[11];
rz(-1.0486773469786526) q[11];
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
ry(3.141311862533634) q[0];
rz(1.0882824030749392) q[0];
ry(0.6213461841207175) q[1];
rz(-1.5725660965004244) q[1];
ry(-1.0592289120320597) q[2];
rz(2.201120375352475) q[2];
ry(-2.276110847265627) q[3];
rz(2.610785190274496) q[3];
ry(2.055090282626318) q[4];
rz(3.1302389147135603) q[4];
ry(1.3626387368435147) q[5];
rz(-2.7011945472965633) q[5];
ry(0.3038031004958857) q[6];
rz(2.935734158272651) q[6];
ry(0.4781641682446905) q[7];
rz(1.5635747961084066) q[7];
ry(0.5594753776838032) q[8];
rz(-0.33544899444304993) q[8];
ry(-0.7078125729300301) q[9];
rz(0.3024854325728623) q[9];
ry(-1.908946801981255) q[10];
rz(3.0730379708459026) q[10];
ry(-1.9400859759416613) q[11];
rz(-1.6151591837996637) q[11];
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
ry(1.5205713912572778) q[0];
rz(0.24796979810311426) q[0];
ry(-2.472665756146639) q[1];
rz(1.5720054849299603) q[1];
ry(0.8492216924651917) q[2];
rz(0.155606815129854) q[2];
ry(-1.2821455785630365) q[3];
rz(-2.79588689424802) q[3];
ry(-0.46827646290015723) q[4];
rz(1.3767455300198035) q[4];
ry(-1.5114929998504267) q[5];
rz(0.25176798123784677) q[5];
ry(-3.1408744380666618) q[6];
rz(-0.20272999469516775) q[6];
ry(0.8035235485631065) q[7];
rz(0.25797239057442356) q[7];
ry(0.8795809978400877) q[8];
rz(1.230714444913426) q[8];
ry(3.0206767576658664) q[9];
rz(2.582016788577026) q[9];
ry(2.5600388460975005) q[10];
rz(0.33519391288087447) q[10];
ry(1.1960251653724387) q[11];
rz(2.1730665440028734) q[11];
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
ry(6.206099905096384e-05) q[0];
rz(2.5601430562766927) q[0];
ry(1.7836161994595114) q[1];
rz(1.1432338834345206) q[1];
ry(-1.5320421334600705) q[2];
rz(0.26000197382412693) q[2];
ry(-0.42661382876530224) q[3];
rz(3.116042081022003) q[3];
ry(1.223044278237654) q[4];
rz(1.7671408459256641) q[4];
ry(-2.1945794659036553) q[5];
rz(2.8512634031942214) q[5];
ry(1.5397677703309665) q[6];
rz(0.9152083106291603) q[6];
ry(0.0001498663182411785) q[7];
rz(3.0199156499391853) q[7];
ry(0.6581291988232483) q[8];
rz(-2.568800067735574) q[8];
ry(1.7447282458689275) q[9];
rz(-2.460568076061632) q[9];
ry(-1.251319133412396) q[10];
rz(-2.057173762193895) q[10];
ry(1.476605408137626) q[11];
rz(-2.7242439707352415) q[11];
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
ry(-1.7798739394929286) q[0];
rz(-2.597948923214157) q[0];
ry(0.8833037535382715) q[1];
rz(0.896655643689291) q[1];
ry(2.6981039169728076) q[2];
rz(2.518945676991668) q[2];
ry(0.9382861159910822) q[3];
rz(1.172561864828829) q[3];
ry(-0.9900022203267582) q[4];
rz(-1.4329397174855585) q[4];
ry(-2.716614742057899) q[5];
rz(1.3691269755532949) q[5];
ry(0.17754375069289186) q[6];
rz(2.4751060172039026) q[6];
ry(2.2130774421510293) q[7];
rz(-1.2338237426105867) q[7];
ry(-2.2082579514985143) q[8];
rz(-0.2700385310102038) q[8];
ry(-0.84398454340712) q[9];
rz(0.6417700251511658) q[9];
ry(1.6665883656833798) q[10];
rz(-0.520392588587112) q[10];
ry(-2.1632459981738097) q[11];
rz(1.8327013634635145) q[11];