OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.5707921572463457) q[0];
rz(3.084115963390657) q[0];
ry(0.0190559305172155) q[1];
rz(-1.5708633038645994) q[1];
ry(-1.5716249646845908) q[2];
rz(3.141590079059755) q[2];
ry(0.0005989328926023774) q[3];
rz(-3.0272857807236675) q[3];
ry(-1.5708073076000417) q[4];
rz(0.9670166210253079) q[4];
ry(3.119413962505818) q[5];
rz(1.5709034738584464) q[5];
ry(-1.626909223831176) q[6];
rz(-0.7854444006024135) q[6];
ry(-0.06699620538828199) q[7];
rz(-1.5706893983123322) q[7];
ry(-1.5711876247225935) q[8];
rz(-1.1761784120696595e-05) q[8];
ry(3.1415924059121) q[9];
rz(2.9956211673645057) q[9];
ry(1.1893869814727855) q[10];
rz(-1.5419094239002276) q[10];
ry(-0.05248947207195222) q[11];
rz(-1.5708001583963722) q[11];
ry(-1.5707990361964228) q[12];
rz(-1.5707950805497537) q[12];
ry(1.7461307308916025) q[13];
rz(-3.2162144627534417e-06) q[13];
ry(-0.32925612029690754) q[14];
rz(-1.5707389066061568) q[14];
ry(1.5799265548400132) q[15];
rz(-0.2039685550383554) q[15];
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
ry(3.1415316539629567) q[0];
rz(2.285479603724685) q[0];
ry(0.040180891454584255) q[1];
rz(1.4858424308052727e-07) q[1];
ry(1.7697259723152645) q[2];
rz(1.7319779919521319) q[2];
ry(3.141412917611093) q[3];
rz(0.4022661151817886) q[3];
ry(3.141590695349524) q[4];
rz(2.5378129831402028) q[4];
ry(-3.0043484303438643) q[5];
rz(3.1415688178841616) q[5];
ry(-3.1397797538205556) q[6];
rz(0.7852468424821444) q[6];
ry(-0.577625338591905) q[7];
rz(1.5102828322177688e-05) q[7];
ry(-0.7149857286166051) q[8];
rz(-2.1662932687955068) q[8];
ry(1.5707962241519493) q[9];
rz(-1.0307614379907835e-05) q[9];
ry(0.005272830639120081) q[10];
rz(-0.028446684990066192) q[10];
ry(0.9182860504122008) q[11];
rz(3.1415450266142013) q[11];
ry(1.5707981018464787) q[12];
rz(0.6569522993278619) q[12];
ry(1.5706818055390483) q[13];
rz(3.105557538952725) q[13];
ry(-0.03885873631405534) q[14];
rz(-0.19901107995872527) q[14];
ry(-3.1403284866408665) q[15];
rz(-0.203966369158809) q[15];
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
ry(8.031805538877279e-06) q[0];
rz(0.5633987701284896) q[0];
ry(-0.8641285713431779) q[1];
rz(-0.11586322151842676) q[1];
ry(-3.114886776382793) q[2];
rz(2.5859602708654736) q[2];
ry(1.5707342001242595) q[3];
rz(-0.5191137452259723) q[3];
ry(1.6150757744018769) q[4];
rz(1.5707962580749257) q[4];
ry(0.6090355535010089) q[5];
rz(-0.5901569935735811) q[5];
ry(-3.126260255494913) q[6];
rz(1.5534582680874602) q[6];
ry(-0.1689222884182447) q[7];
rz(2.0294300531243152) q[7];
ry(-0.007654212394681253) q[8];
rz(-0.3401771560888572) q[8];
ry(-3.034258481459135) q[9];
rz(0.5025308654246519) q[9];
ry(0.018373455202487676) q[10];
rz(2.193124567822413) q[10];
ry(3.0746392552170447) q[11];
rz(-3.044625389823888) q[11];
ry(1.5707963023442644) q[12];
rz(1.5710251693209154) q[12];
ry(1.5720138503185983) q[13];
rz(-1.5708697534745786) q[13];
ry(-0.0004195608381110105) q[14];
rz(2.276643437267542) q[14];
ry(-1.5707891066134878) q[15];
rz(0.06746793327645673) q[15];
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
ry(5.8152903825714475e-08) q[0];
rz(-1.6354835406714283) q[0];
ry(-1.6569260923304796e-06) q[1];
rz(1.9737580896647973) q[1];
ry(-5.757061760647275e-08) q[2];
rz(-1.4152170977152636) q[2];
ry(1.225542365368426e-06) q[3];
rz(2.6822385043920374) q[3];
ry(-1.5707882345727648) q[4];
rz(-0.6791561231493981) q[4];
ry(8.229513039914593e-06) q[5];
rz(1.759852877673719) q[5];
ry(3.1415800908614844) q[6];
rz(1.6940225908884583) q[6];
ry(-6.769727802868887e-05) q[7];
rz(1.9197718338183318) q[7];
ry(-1.4092440991930744e-08) q[8];
rz(-1.5635139699265048) q[8];
ry(-3.1411342227620818) q[9];
rz(0.5025419921480996) q[9];
ry(1.2475395210934212e-06) q[10];
rz(-2.8709130792643913) q[10];
ry(-3.1415876463253527) q[11];
rz(0.6148810390553844) q[11];
ry(-3.1414590111095686) q[12];
rz(-1.5705683382729674) q[12];
ry(-1.5708124596583761) q[13];
rz(1.5707918445921825) q[13];
ry(3.1415884161776577) q[14];
rz(-2.2725682942965397) q[14];
ry(1.5707954389697174) q[15];
rz(-1.5709964731160717) q[15];
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
ry(3.1398586746144175) q[0];
rz(-0.3001910630523934) q[0];
ry(1.7293177734245546e-07) q[1];
rz(-0.2936488259213781) q[1];
ry(8.343940278995183e-06) q[2];
rz(1.9140686409438008) q[2];
ry(-3.1415924734693736) q[3];
rz(2.1871319775811715) q[3];
ry(3.1398698975564656) q[4];
rz(-0.6791362449708869) q[4];
ry(2.5214026319119496e-07) q[5];
rz(0.39854772504141095) q[5];
ry(-3.141063857736911) q[6];
rz(0.1405445420014878) q[6];
ry(-6.532012310283805e-07) q[7];
rz(-2.379047553459146) q[7];
ry(-2.3309105207908415e-05) q[8];
rz(0.9285356892603269) q[8];
ry(1.5707959661547992) q[9];
rz(-1.5708791575486147) q[9];
ry(-1.370426269150449e-05) q[10];
rz(-0.8934301938894329) q[10];
ry(-1.0199630693079968e-06) q[11];
rz(-2.0889705030323444) q[11];
ry(1.5710186292582708) q[12];
rz(-2.2346593145243876) q[12];
ry(1.5709969407829076) q[13];
rz(1.6146887741764426) q[13];
ry(3.0665385157327316e-06) q[14];
rz(-0.3621330976344001) q[14];
ry(1.5708009049881246) q[15];
rz(1.6546731876879832) q[15];
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
ry(-1.5706883754857586) q[0];
rz(1.1847990695044768) q[0];
ry(-3.123534877062311) q[1];
rz(-2.5477947625360757) q[1];
ry(1.3488428396571637) q[2];
rz(0.7885963387906553) q[2];
ry(1.570950413590083) q[3];
rz(3.1240599726279026) q[3];
ry(-1.5282260707715165) q[4];
rz(-2.772592560969498) q[4];
ry(3.1203928020686593) q[5];
rz(0.16121278792623106) q[5];
ry(-1.5709183252472798) q[6];
rz(1.466901241337211) q[6];
ry(0.057465030652910976) q[7];
rz(-1.6356783980129554) q[7];
ry(-2.562118593993905) q[8];
rz(1.451084306134252) q[8];
ry(-1.6890919850147377) q[9];
rz(1.874422832369749) q[9];
ry(1.1711819692380367) q[10];
rz(-1.5770181075638314) q[10];
ry(-0.13919567382270506) q[11];
rz(2.1436005245344054) q[11];
ry(-3.1415891909770712) q[12];
rz(-2.9020348198324983) q[12];
ry(-3.125912602996193) q[13];
rz(1.4788576560811117) q[13];
ry(1.7866404494253776) q[14];
rz(2.153479478376733) q[14];
ry(-0.0007003207859354141) q[15];
rz(2.70659268780664) q[15];
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
ry(-1.5711481427033647) q[0];
rz(-3.070338242906079) q[0];
ry(-3.141306837587916) q[1];
rz(2.237890619672287) q[1];
ry(-1.7846398241850647) q[2];
rz(-3.140856026267886) q[2];
ry(0.009104810452806965) q[3];
rz(1.3730622722302346) q[3];
ry(-1.1699289721001094e-05) q[4];
rz(-1.9397993076643978) q[4];
ry(-7.853766400778284e-06) q[5];
rz(0.7733186567834193) q[5];
ry(3.141589492200651) q[6];
rz(-2.311311241685405) q[6];
ry(-3.1414271917095427) q[7];
rz(-1.9679137672731033) q[7];
ry(-1.3201430680389254) q[8];
rz(-1.2459724248564794) q[8];
ry(-3.141323034766487) q[9];
rz(2.2451464525839206) q[9];
ry(0.004715224796494526) q[10];
rz(1.5920082471521753) q[10];
ry(-3.1415227831293437) q[11];
rz(2.1698519071841815) q[11];
ry(3.1415199739625193) q[12];
rz(-2.236213538914239) q[12];
ry(3.1415448242177595) q[13];
rz(-2.628917491407887) q[13];
ry(-3.1413431148466313) q[14];
rz(-1.4760525993369757) q[14];
ry(1.896777186924483e-05) q[15];
rz(2.60757737576776) q[15];
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
ry(0.0012934919720466453) q[0];
rz(3.0703405946409283) q[0];
ry(-3.1415901841449028) q[1];
rz(1.1065183538095935) q[1];
ry(1.5695958217390444) q[2];
rz(-3.141591980563923) q[2];
ry(-6.352396315634223e-07) q[3];
rz(2.550437144505738) q[3];
ry(-1.2866755330400352) q[4];
rz(-3.1415769999398027) q[4];
ry(7.454196877779834e-07) q[5];
rz(-0.02172739966523185) q[5];
ry(1.5707995201310552) q[6];
rz(-3.1415766542235963) q[6];
ry(3.1415921287528485) q[7];
rz(-1.3601207312239625) q[7];
ry(1.570794328448656) q[8];
rz(-1.5707898422468445) q[8];
ry(-3.141592156468571) q[9];
rz(1.5467565769424967) q[9];
ry(5.992270267034194e-06) q[10];
rz(0.5713535805955284) q[10];
ry(-4.4462750200473663e-07) q[11];
rz(-1.2745601078070756) q[11];
ry(0.01903252566337965) q[12];
rz(-1.9849915835783758) q[12];
ry(-3.1415890123136814) q[13];
rz(-0.4411391830011316) q[13];
ry(3.1415773297326397) q[14];
rz(1.1459980245944352) q[14];
ry(-3.14158859008952) q[15];
rz(0.21762140495146604) q[15];
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
ry(-1.5707688786646932) q[0];
rz(-5.391715380831386e-06) q[0];
ry(-2.502036554162124e-07) q[1];
rz(1.6892736154903443) q[1];
ry(-1.5708241360883843) q[2];
rz(-3.141587035050486) q[2];
ry(3.430369588386384e-08) q[3];
rz(-0.7252537106333241) q[3];
ry(-1.5707913153366324) q[4];
rz(-0.08012020897515183) q[4];
ry(-3.1415924941275772) q[5];
rz(2.0701623456808598) q[5];
ry(-1.5707850500905847) q[6];
rz(-3.1415497667640686) q[6];
ry(7.84231280756595e-08) q[7];
rz(-3.0070546019881705) q[7];
ry(1.5707900900907417) q[8];
rz(2.328936849550702) q[8];
ry(1.2450731112778612e-08) q[9];
rz(0.09046179409891762) q[9];
ry(1.0438506208743092e-05) q[10];
rz(-1.7277821695067157) q[10];
ry(-3.1415925212162343) q[11];
rz(0.30228584848773293) q[11];
ry(3.141590127237564) q[12];
rz(2.7293501472505795) q[12];
ry(-3.141584337065865) q[13];
rz(2.094850434510148) q[13];
ry(-2.3416007289899137e-06) q[14];
rz(-0.7122909800214572) q[14];
ry(-8.245828628794527e-06) q[15];
rz(-1.0642539573765573) q[15];
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
ry(-1.5707414753330013) q[0];
rz(3.0736374955021977) q[0];
ry(-3.141591496835402) q[1];
rz(-0.4802932012338905) q[1];
ry(-1.5708497590436492) q[2];
rz(-0.06791048290446228) q[2];
ry(1.8419854512785605e-06) q[3];
rz(-0.10686315868655338) q[3];
ry(3.141526056727336) q[4];
rz(-0.14793729087419888) q[4];
ry(-1.4358260593473132e-07) q[5];
rz(-2.7933490337210567) q[5];
ry(-1.5707947216986708) q[6];
rz(1.5028171745202994) q[6];
ry(3.1415926215579333) q[7];
rz(-2.5325415451689968) q[7];
ry(3.1415910038355666) q[8];
rz(-0.8804773507435861) q[8];
ry(-3.1415915033721) q[9];
rz(-0.3720766592176083) q[9];
ry(-3.1415853799480606) q[10];
rz(-2.780033642041044) q[10];
ry(3.141591233891138) q[11];
rz(3.053298213555602) q[11];
ry(1.570789861200935) q[12];
rz(3.073927048195671) q[12];
ry(3.1415734007946807) q[13];
rz(3.1167397630789964) q[13];
ry(3.141582984335176) q[14];
rz(2.424758727419704) q[14];
ry(3.1415732603601327) q[15];
rz(3.1123393597417977) q[15];