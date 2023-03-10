OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.8357987762747173) q[0];
rz(2.924381044253326) q[0];
ry(2.7784857160233862) q[1];
rz(0.32122521208993454) q[1];
ry(2.8107248839140238) q[2];
rz(-2.0413616375888566) q[2];
ry(2.479784501570023) q[3];
rz(2.488228106391552) q[3];
ry(-1.390552166107076) q[4];
rz(-0.10334717825851403) q[4];
ry(2.5740422023192004) q[5];
rz(2.7458083261877944) q[5];
ry(-0.868456939382013) q[6];
rz(2.788214275311227) q[6];
ry(0.8159613056738089) q[7];
rz(-2.231930021498336) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.538186922458363) q[0];
rz(-2.930083692050345) q[0];
ry(2.9097014322904915) q[1];
rz(2.672347405564548) q[1];
ry(0.9170391563635214) q[2];
rz(1.176049739715186) q[2];
ry(1.1234369520180545) q[3];
rz(2.2004945100006) q[3];
ry(-2.770569363922963) q[4];
rz(-1.5632024580606965) q[4];
ry(-1.8325776271437189) q[5];
rz(-1.5475484100598091) q[5];
ry(-2.862881966201956) q[6];
rz(2.739510171630181) q[6];
ry(2.0534993868228075) q[7];
rz(-1.9435000446991284) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.4777635164074025) q[0];
rz(-2.8347732766561307) q[0];
ry(0.38618988625414197) q[1];
rz(-3.0379220656999957) q[1];
ry(-2.362437489077085) q[2];
rz(2.617788671995591) q[2];
ry(-2.158563661766016) q[3];
rz(-2.0748762619789076) q[3];
ry(-1.034303967454755) q[4];
rz(0.5199066571559507) q[4];
ry(1.531366104955361) q[5];
rz(2.9349929019005807) q[5];
ry(0.8895380666193722) q[6];
rz(-2.33080611532768) q[6];
ry(-2.6345380887014347) q[7];
rz(0.45265421595210886) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.7866300025542037) q[0];
rz(-2.787661195162386) q[0];
ry(-2.207603882514672) q[1];
rz(-0.5080331679573935) q[1];
ry(-1.4737136136849924) q[2];
rz(0.9516225330720164) q[2];
ry(-1.1642559353057065) q[3];
rz(-1.7324835517130073) q[3];
ry(2.881287327245357) q[4];
rz(-1.6483566660444937) q[4];
ry(-0.16092789619239412) q[5];
rz(-2.884542123879863) q[5];
ry(0.7368174277466222) q[6];
rz(-1.9701228481886037) q[6];
ry(-2.385200977849946) q[7];
rz(-0.8035617786996263) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.09432375199592866) q[0];
rz(-2.1738884553111166) q[0];
ry(0.5424261969291768) q[1];
rz(-3.105890516998937) q[1];
ry(-0.5895900038724706) q[2];
rz(-0.6289351385546712) q[2];
ry(-1.638578233267934) q[3];
rz(1.7283758147051296) q[3];
ry(-2.81893407544897) q[4];
rz(-0.42682869606761553) q[4];
ry(0.1588381004012671) q[5];
rz(-1.9796786346107282) q[5];
ry(1.4983105399218128) q[6];
rz(-0.37817030727222467) q[6];
ry(-0.7077672881145677) q[7];
rz(-0.3591959663245811) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.8925326864187815) q[0];
rz(-2.9393424932890513) q[0];
ry(-2.9140809171720594) q[1];
rz(-1.4297940834112737) q[1];
ry(1.355972265299143) q[2];
rz(-1.8440396084293873) q[2];
ry(1.4226166085443694) q[3];
rz(2.7076361110243083) q[3];
ry(-2.340886540982456) q[4];
rz(-2.7982845562655103) q[4];
ry(-2.978668023916259) q[5];
rz(-0.7270589440535948) q[5];
ry(-0.3030059475960547) q[6];
rz(2.065726370913659) q[6];
ry(-1.5733507783799472) q[7];
rz(-0.25752922909335657) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.16647574869783) q[0];
rz(-0.1740610822171238) q[0];
ry(0.9400204404682193) q[1];
rz(-1.1591460084922136) q[1];
ry(-1.5478333965742443) q[2];
rz(-1.8174900490174306) q[2];
ry(0.9517939673313419) q[3];
rz(-0.2756363674151183) q[3];
ry(2.680457347373107) q[4];
rz(2.7571894138772954) q[4];
ry(-0.8064450773641773) q[5];
rz(-0.9061989189164761) q[5];
ry(-0.6109549796123446) q[6];
rz(1.936801233070999) q[6];
ry(2.5985179485421686) q[7];
rz(-1.4578588940961845) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.0398017435766267) q[0];
rz(2.9878822895148454) q[0];
ry(2.5011515459431335) q[1];
rz(2.310300881239747) q[1];
ry(-2.597993841405125) q[2];
rz(-2.226953948630873) q[2];
ry(-1.6403815383040627) q[3];
rz(3.0613668247266275) q[3];
ry(0.8926022242351128) q[4];
rz(0.945910909865816) q[4];
ry(2.34513417172295) q[5];
rz(0.30531919040374333) q[5];
ry(0.7865975610703203) q[6];
rz(3.014509001200742) q[6];
ry(-1.930746271643166) q[7];
rz(3.0216447773314563) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.166743777541168) q[0];
rz(-0.9552775673855552) q[0];
ry(1.3712336245882215) q[1];
rz(-1.6109484201230382) q[1];
ry(-0.7807593502674214) q[2];
rz(-2.1449647223972486) q[2];
ry(2.1468057733038215) q[3];
rz(0.906700868025662) q[3];
ry(0.5904860542490539) q[4];
rz(-2.0586109024269468) q[4];
ry(0.32807786054069016) q[5];
rz(-2.4242402873630775) q[5];
ry(2.367030992685091) q[6];
rz(1.356907555995595) q[6];
ry(2.1851800975853495) q[7];
rz(0.4147052297605258) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.9900108264350636) q[0];
rz(0.4755528162060863) q[0];
ry(1.624333908143794) q[1];
rz(-1.600400829453779) q[1];
ry(1.5845188696816777) q[2];
rz(-2.710881384874871) q[2];
ry(0.4241256039487524) q[3];
rz(-0.9231730358577247) q[3];
ry(-1.9876137581308946) q[4];
rz(0.1773080809917771) q[4];
ry(2.7459922170738453) q[5];
rz(-0.41904902950534334) q[5];
ry(0.9080736686483606) q[6];
rz(1.816018093023871) q[6];
ry(-2.31910272773545) q[7];
rz(-1.5271839488956587) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.3377436195898245) q[0];
rz(2.5379238017778905) q[0];
ry(3.105341939638293) q[1];
rz(-2.0545437229595143) q[1];
ry(-1.6264223408970144) q[2];
rz(2.1978100072363427) q[2];
ry(1.6591900731727467) q[3];
rz(-1.3601386550193526) q[3];
ry(2.228454246495134) q[4];
rz(-1.8923994256340562) q[4];
ry(1.0676788787890334) q[5];
rz(1.8498485593105551) q[5];
ry(-0.5666034531319409) q[6];
rz(-1.1734495712463207) q[6];
ry(0.22655692757398338) q[7];
rz(1.9046991967977043) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.4185979615500903) q[0];
rz(-1.0106769487221205) q[0];
ry(-1.9664692262676766) q[1];
rz(2.8323757403901166) q[1];
ry(-1.458839454921325) q[2];
rz(-1.4804780190452873) q[2];
ry(0.6902880773821415) q[3];
rz(2.970711999232629) q[3];
ry(1.5472173455074296) q[4];
rz(-1.4476211580714224) q[4];
ry(-0.9190081025549288) q[5];
rz(0.5725852130700487) q[5];
ry(-0.3563879237250785) q[6];
rz(1.5754809560339256) q[6];
ry(2.7412606236785195) q[7];
rz(-1.0212366422787362) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.7991066134633833) q[0];
rz(2.9353625061824067) q[0];
ry(2.180472086115758) q[1];
rz(0.5828494535769169) q[1];
ry(-0.40650114974774204) q[2];
rz(1.8064840485017983) q[2];
ry(-0.7567739495153906) q[3];
rz(-0.34645585814527874) q[3];
ry(1.6609827894879097) q[4];
rz(-1.3702373079802808) q[4];
ry(-1.7662955872713881) q[5];
rz(0.3516182334699902) q[5];
ry(1.9579893455861668) q[6];
rz(2.3987106417262685) q[6];
ry(-1.1788804045428867) q[7];
rz(-1.258903501038949) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.7983262388182378) q[0];
rz(-2.3637858347175986) q[0];
ry(1.308585910440511) q[1];
rz(-2.824258957355804) q[1];
ry(-2.6592947205501294) q[2];
rz(0.3575554216714134) q[2];
ry(-2.040752407139315) q[3];
rz(0.7529655629252482) q[3];
ry(-1.6508632872644045) q[4];
rz(2.248946694292494) q[4];
ry(0.22119785349415455) q[5];
rz(0.38071363364707855) q[5];
ry(2.3857242476012286) q[6];
rz(1.1459696487936286) q[6];
ry(3.0986868701484935) q[7];
rz(1.8274093451265854) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.9402570495975692) q[0];
rz(1.30165982527024) q[0];
ry(-1.5786813248035303) q[1];
rz(-0.7100581210002963) q[1];
ry(0.28645804827004734) q[2];
rz(-1.0483386840723672) q[2];
ry(1.6173868462927632) q[3];
rz(0.2288077920835523) q[3];
ry(-2.4237607378201265) q[4];
rz(-1.8862645360413255) q[4];
ry(2.5460650353710697) q[5];
rz(0.0015215513716571039) q[5];
ry(0.559012436800795) q[6];
rz(2.4130990583264684) q[6];
ry(2.2707674586644684) q[7];
rz(2.0487400668721376) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.544350814276446) q[0];
rz(-1.4789804266228994) q[0];
ry(0.10990201506199737) q[1];
rz(-0.8419966178120246) q[1];
ry(0.35988462970813373) q[2];
rz(2.0120834449118474) q[2];
ry(0.5633874244407213) q[3];
rz(0.45412679837550307) q[3];
ry(2.4226714971941026) q[4];
rz(-2.528106911493089) q[4];
ry(-2.118394418131365) q[5];
rz(1.2017262352220133) q[5];
ry(2.2408607836763395) q[6];
rz(-0.6732640115046333) q[6];
ry(1.2817072210990328) q[7];
rz(-0.7600621087462338) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.7492023225669753) q[0];
rz(-1.8367672680200355) q[0];
ry(-2.560843682367285) q[1];
rz(0.863401980965336) q[1];
ry(2.632602051268374) q[2];
rz(0.11366874403578696) q[2];
ry(-1.2419157796234153) q[3];
rz(-0.04968372157781114) q[3];
ry(-1.1586755980490702) q[4];
rz(0.21647133516352748) q[4];
ry(1.7036358650641867) q[5];
rz(-1.3743859823091054) q[5];
ry(-1.3809662429357612) q[6];
rz(-0.5746500996715779) q[6];
ry(-1.718221634158735) q[7];
rz(-2.8467270242467575) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.6234229542872685) q[0];
rz(-1.859288821198393) q[0];
ry(-2.422797899046856) q[1];
rz(-2.950027120688818) q[1];
ry(0.37953596734878703) q[2];
rz(-0.3153286056914401) q[2];
ry(2.3223431479893986) q[3];
rz(2.0650334420789127) q[3];
ry(-3.042936291027157) q[4];
rz(-0.09051874843415142) q[4];
ry(-0.5687285675031452) q[5];
rz(0.863233391068194) q[5];
ry(-0.12092777430308921) q[6];
rz(-1.3058389305354225) q[6];
ry(1.327589038182258) q[7];
rz(-2.0607135729646555) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.6135675492820528) q[0];
rz(0.42228147342591765) q[0];
ry(-0.3955688397898278) q[1];
rz(-2.7623774125094607) q[1];
ry(-2.0925426135153584) q[2];
rz(-1.0568640292046594) q[2];
ry(-0.9021319994357587) q[3];
rz(1.271519709711379) q[3];
ry(-0.26429196589952936) q[4];
rz(-1.1843904083365517) q[4];
ry(0.2444153349829782) q[5];
rz(2.922514670057682) q[5];
ry(-1.9601294567842826) q[6];
rz(1.4186071689851845) q[6];
ry(-1.1579455596256185) q[7];
rz(-1.8777464718185515) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.7992840410831423) q[0];
rz(2.2364415744459167) q[0];
ry(0.08095302415974448) q[1];
rz(-1.7838828981489474) q[1];
ry(-2.6270441521302508) q[2];
rz(-1.5436826945646223) q[2];
ry(2.265550419215094) q[3];
rz(-1.8289864706623555) q[3];
ry(-0.70752455794325) q[4];
rz(1.6929997632264806) q[4];
ry(-0.4122335177027787) q[5];
rz(-0.6926877359527017) q[5];
ry(1.7956009413209464) q[6];
rz(-3.002485474638028) q[6];
ry(-0.7807633506202949) q[7];
rz(2.6473256449923164) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.0834773848453034) q[0];
rz(0.12503484550685687) q[0];
ry(-0.3633695177685535) q[1];
rz(-1.4160579051205766) q[1];
ry(1.1595637183693137) q[2];
rz(2.758056561100366) q[2];
ry(-0.5813882970558295) q[3];
rz(2.54281066880843) q[3];
ry(-1.5564427500587277) q[4];
rz(-1.4076883623941) q[4];
ry(-2.3701662214950305) q[5];
rz(1.6744790781324428) q[5];
ry(-0.8132337085126102) q[6];
rz(-1.5327571611299726) q[6];
ry(3.000121727875509) q[7];
rz(-2.897399893912362) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.3299938047170854) q[0];
rz(-1.7350529052156571) q[0];
ry(0.3864027543254558) q[1];
rz(1.677832445950685) q[1];
ry(2.7461527712213822) q[2];
rz(2.3855860036537044) q[2];
ry(-2.1210695683442875) q[3];
rz(1.0726776051090443) q[3];
ry(1.6297537941114795) q[4];
rz(0.9472880170509708) q[4];
ry(-2.567625846151584) q[5];
rz(-2.420386161470353) q[5];
ry(-0.8329002081591304) q[6];
rz(1.8679468015403593) q[6];
ry(1.895819507270036) q[7];
rz(2.335691069999948) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.963110052809867) q[0];
rz(-0.5130979760189476) q[0];
ry(0.6345644959216958) q[1];
rz(1.2037723571419003) q[1];
ry(1.3815020678320602) q[2];
rz(0.5728976194436592) q[2];
ry(1.4579747208318095) q[3];
rz(1.1436171100700878) q[3];
ry(-2.4834282713276146) q[4];
rz(-2.238959754618353) q[4];
ry(-1.282662462543719) q[5];
rz(2.0212459827565645) q[5];
ry(-2.1648584050668553) q[6];
rz(2.928448125671591) q[6];
ry(0.535917023693365) q[7];
rz(2.842800670616188) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.0893548746576793) q[0];
rz(-2.10614560322777) q[0];
ry(-0.9122132921930826) q[1];
rz(2.4678451520040148) q[1];
ry(-0.8939654967249702) q[2];
rz(2.6577564593074423) q[2];
ry(-2.671576774006489) q[3];
rz(1.3871250461195137) q[3];
ry(-0.20982046269649526) q[4];
rz(-2.5467192920979804) q[4];
ry(2.839114265043353) q[5];
rz(2.457665521177377) q[5];
ry(1.2384801681702444) q[6];
rz(1.2718059010205858) q[6];
ry(-0.3657043022629929) q[7];
rz(-2.4010699204190473) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.1061107351760622) q[0];
rz(1.1614270864061513) q[0];
ry(-1.6240704625895215) q[1];
rz(-1.2894710327269578) q[1];
ry(-1.9324876576186956) q[2];
rz(-2.5133509912743777) q[2];
ry(2.678377829242342) q[3];
rz(1.651725335202545) q[3];
ry(0.38024976501229) q[4];
rz(-0.4350860639543369) q[4];
ry(0.11760606161683371) q[5];
rz(0.4611200983216332) q[5];
ry(-1.7034412451850471) q[6];
rz(2.0887911056289497) q[6];
ry(-1.5608618920437827) q[7];
rz(-2.070859884266639) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.654571120624443) q[0];
rz(2.40938967976713) q[0];
ry(1.261543704810654) q[1];
rz(-2.835255767086115) q[1];
ry(-2.864117809936487) q[2];
rz(1.935093449718564) q[2];
ry(-0.269665432208627) q[3];
rz(2.1859683448602105) q[3];
ry(0.5544781161160585) q[4];
rz(1.3574968075332006) q[4];
ry(1.7460241863572543) q[5];
rz(1.952034747023509) q[5];
ry(-0.6281682610661141) q[6];
rz(0.3191274186749) q[6];
ry(2.339165137157719) q[7];
rz(-1.734565686227066) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.6972172219208845) q[0];
rz(0.2383838040430657) q[0];
ry(2.04757333372556) q[1];
rz(-2.2754709657133967) q[1];
ry(-2.0012703571774924) q[2];
rz(-2.203059423211862) q[2];
ry(1.0776521868484181) q[3];
rz(-0.7351990147256227) q[3];
ry(2.4029289918556382) q[4];
rz(1.2505148128284211) q[4];
ry(-0.6285732108541585) q[5];
rz(1.071784398766984) q[5];
ry(-0.5562644401627956) q[6];
rz(-1.2363488559796603) q[6];
ry(-3.0239363582131427) q[7];
rz(0.6180358089954916) q[7];