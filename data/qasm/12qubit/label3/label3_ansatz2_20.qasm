OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.6054861441780583) q[0];
rz(1.7827334602128238) q[0];
ry(-0.28530207706003724) q[1];
rz(0.15258006962097376) q[1];
ry(-0.19961652387887432) q[2];
rz(-1.3202929487946728) q[2];
ry(-0.7596935698186601) q[3];
rz(-0.9515045263082147) q[3];
ry(-0.842817740266024) q[4];
rz(-1.826730427976134) q[4];
ry(1.545181280018733) q[5];
rz(-1.5803760489629395) q[5];
ry(-1.7098954899916121) q[6];
rz(-2.17200723568845) q[6];
ry(-1.5411722194098811) q[7];
rz(-1.5735648756879133) q[7];
ry(2.4626421515636556) q[8];
rz(0.9881992358466453) q[8];
ry(2.6437582737097576) q[9];
rz(2.525232351983104) q[9];
ry(-1.9438173220190202) q[10];
rz(-2.1975118704761147) q[10];
ry(-0.4333291882200694) q[11];
rz(-2.1478726972060276) q[11];
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
ry(-1.3458405969063294) q[0];
rz(2.9883965301065) q[0];
ry(2.102139525826651) q[1];
rz(-2.776348604753062) q[1];
ry(0.9185158830374317) q[2];
rz(0.11824638161828514) q[2];
ry(1.398126449179115) q[3];
rz(-2.428195284821259) q[3];
ry(1.8053098454681227) q[4];
rz(-3.1017300073703575) q[4];
ry(-1.5556906595573308) q[5];
rz(-2.512840738449587) q[5];
ry(0.6774796681397026) q[6];
rz(-0.8151347130278981) q[6];
ry(-1.5729008171365404) q[7];
rz(-0.31455093120809435) q[7];
ry(0.49649354464782824) q[8];
rz(2.193808747305863) q[8];
ry(2.5496277896998607) q[9];
rz(2.6082529734971747) q[9];
ry(-2.3005569069831484) q[10];
rz(-2.610056720913628) q[10];
ry(-2.463043818532115) q[11];
rz(2.2417040674841435) q[11];
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
ry(-1.558182448585982) q[0];
rz(-2.725241339955373) q[0];
ry(-0.7893555496594394) q[1];
rz(-1.9259281456196575) q[1];
ry(-2.7671326167755965) q[2];
rz(1.6519515593656067) q[2];
ry(-2.7921478116196807) q[3];
rz(-0.6961864403839835) q[3];
ry(0.6219972320117909) q[4];
rz(0.2781363889866616) q[4];
ry(-0.0017341458132723775) q[5];
rz(-0.8690505636655096) q[5];
ry(-1.3532046715838035) q[6];
rz(0.5299666459454978) q[6];
ry(-3.1086401570366764) q[7];
rz(-2.8120298760235944) q[7];
ry(-2.153512566566577) q[8];
rz(1.773271140666135) q[8];
ry(-2.3227380525065557) q[9];
rz(-1.7318793241296477) q[9];
ry(-2.5485999831777018) q[10];
rz(-1.0454202946135291) q[10];
ry(-2.284882218660927) q[11];
rz(2.5102300437246092) q[11];
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
ry(-2.474083518616041) q[0];
rz(-2.6315284151173985) q[0];
ry(0.7707644836394886) q[1];
rz(1.1307765589905152) q[1];
ry(2.3757057855859736) q[2];
rz(-0.1413178850703916) q[2];
ry(1.1551513759097127) q[3];
rz(2.8044813623303755) q[3];
ry(0.6118457173223403) q[4];
rz(0.4954044934683299) q[4];
ry(1.574138109831915) q[5];
rz(-0.03726673273956283) q[5];
ry(-1.451853223561475) q[6];
rz(0.36798308622972714) q[6];
ry(-1.5559625329049345) q[7];
rz(-2.3774216188366304) q[7];
ry(2.988859522084419) q[8];
rz(1.5998626837863144) q[8];
ry(0.8317430148074604) q[9];
rz(-2.96922968895159) q[9];
ry(2.4209183969163135) q[10];
rz(-2.4290779397420486) q[10];
ry(1.8870652316882133) q[11];
rz(-1.4206530862249789) q[11];
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
ry(1.6255215838724641) q[0];
rz(1.7729105828343297) q[0];
ry(-0.39881809623201336) q[1];
rz(1.4593499011854079) q[1];
ry(2.188092725321834) q[2];
rz(2.355711650339256) q[2];
ry(-2.423208437593708) q[3];
rz(1.2911199528941353) q[3];
ry(-1.9843917567762541) q[4];
rz(-1.864738434711664) q[4];
ry(-2.579478612895043) q[5];
rz(-2.4565773842259953) q[5];
ry(2.298421625777029) q[6];
rz(-0.2241670994702005) q[6];
ry(3.136543429984588) q[7];
rz(2.3329933180124094) q[7];
ry(0.3707755163395783) q[8];
rz(-0.20188874016938385) q[8];
ry(-0.10667352806670838) q[9];
rz(2.2363679141582633) q[9];
ry(-2.5189851245682036) q[10];
rz(-1.175392086378579) q[10];
ry(-2.1975991434511717) q[11];
rz(2.535506115321147) q[11];
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
ry(-0.947822542040872) q[0];
rz(-3.1374822294632665) q[0];
ry(1.8714912762580544) q[1];
rz(-1.8064488738472564) q[1];
ry(1.6987353592477106) q[2];
rz(-0.6577616387637928) q[2];
ry(-2.797763971841328) q[3];
rz(-2.20045776895408) q[3];
ry(0.5187483793774641) q[4];
rz(2.7646645792589926) q[4];
ry(-3.131649878001801) q[5];
rz(0.7244289805178523) q[5];
ry(-1.8462694697789985) q[6];
rz(2.2502839626822952) q[6];
ry(-1.0865080265064906) q[7];
rz(1.5967072040421417) q[7];
ry(-1.7231039392982332) q[8];
rz(0.12915760142056099) q[8];
ry(-1.6125013960744097) q[9];
rz(0.5618139566301257) q[9];
ry(-1.3031885916202386) q[10];
rz(-1.5285913723315048) q[10];
ry(-0.5330832906621019) q[11];
rz(2.419640930534902) q[11];
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
ry(-2.8082806912739278) q[0];
rz(0.42861824690467554) q[0];
ry(1.7761357566574818) q[1];
rz(0.4794667479214194) q[1];
ry(0.34702432548529305) q[2];
rz(-0.5754711799795251) q[2];
ry(1.38122854329256) q[3];
rz(0.3396853354348688) q[3];
ry(2.579961464236759) q[4];
rz(-1.8867516800830282) q[4];
ry(-1.5039651378111056) q[5];
rz(1.6858226503515867) q[5];
ry(1.729224678894384) q[6];
rz(-2.9300210924924857) q[6];
ry(2.120137179877081) q[7];
rz(-0.4369747458576238) q[7];
ry(-0.5433927491748068) q[8];
rz(-1.5606614714525096) q[8];
ry(1.5960214201296459) q[9];
rz(-1.316324537612522) q[9];
ry(-1.5772615734915982) q[10];
rz(0.8415180638774159) q[10];
ry(0.3160453837909838) q[11];
rz(-0.027174111902177143) q[11];
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
ry(-1.9080638457855716) q[0];
rz(-0.32376075766081774) q[0];
ry(0.5708085922958377) q[1];
rz(-2.647735993539287) q[1];
ry(-1.452654218530271) q[2];
rz(1.1024695241382163) q[2];
ry(-0.5075889075204115) q[3];
rz(-1.8144635490046885) q[3];
ry(-2.3500798472505795) q[4];
rz(-0.779971287348973) q[4];
ry(2.989556655872184) q[5];
rz(1.6575880771837155) q[5];
ry(2.5514973387779842) q[6];
rz(2.675915136766936) q[6];
ry(-3.1396329145339084) q[7];
rz(0.5450867219527306) q[7];
ry(-1.6708331806043106) q[8];
rz(-0.4386157660488324) q[8];
ry(-2.996054094822857) q[9];
rz(-1.5919008463292075) q[9];
ry(0.24777682323576894) q[10];
rz(1.20793389162029) q[10];
ry(-0.5157205426141658) q[11];
rz(1.7928464831042028) q[11];
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
ry(-1.3522466798907686) q[0];
rz(3.12424045204197) q[0];
ry(1.5789410394025525) q[1];
rz(-0.9618904528911525) q[1];
ry(1.9387297617023522) q[2];
rz(1.5911678076204352) q[2];
ry(-1.6730444090703553) q[3];
rz(0.7944881639811175) q[3];
ry(-1.0401293202358035) q[4];
rz(1.9747131894433811) q[4];
ry(-2.6746907063999914) q[5];
rz(1.5268211753837644) q[5];
ry(-1.1994398319135575) q[6];
rz(1.161565053833408) q[6];
ry(-3.127392266797901) q[7];
rz(2.5644575374792096) q[7];
ry(2.146541452406776) q[8];
rz(-2.379939355869204) q[8];
ry(-0.8884505490266603) q[9];
rz(-0.22740541380638657) q[9];
ry(1.7048536267581602) q[10];
rz(2.7510346272014217) q[10];
ry(0.8760965675630368) q[11];
rz(-2.647969648951877) q[11];
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
ry(-1.59375229530312) q[0];
rz(-1.6437183532536501) q[0];
ry(2.5240052188637723) q[1];
rz(2.6956891360398156) q[1];
ry(-2.3187811078704814) q[2];
rz(2.02809694807711) q[2];
ry(-2.5570120061475734) q[3];
rz(-0.569973580665418) q[3];
ry(-0.8841491967713857) q[4];
rz(-1.2640130780733136) q[4];
ry(1.5681925952452094) q[5];
rz(1.5596143244474066) q[5];
ry(1.3201328874473586) q[6];
rz(-1.6037606383989669) q[6];
ry(-0.316457879966794) q[7];
rz(1.5743387198866197) q[7];
ry(2.0784901006093737) q[8];
rz(2.6835992720620174) q[8];
ry(-0.2727376181310719) q[9];
rz(2.7761096340429954) q[9];
ry(2.537837926501391) q[10];
rz(0.017174125888349843) q[10];
ry(0.929729117617339) q[11];
rz(-2.582153479551588) q[11];
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
ry(-1.114612423916838) q[0];
rz(-0.8488178162775801) q[0];
ry(2.1277097159707687) q[1];
rz(1.4537417726136532) q[1];
ry(-1.4829061675526471) q[2];
rz(-2.0447737227828204) q[2];
ry(-2.6239205984611074) q[3];
rz(-1.7684643879021296) q[3];
ry(2.517126763191446) q[4];
rz(2.898634572214721) q[4];
ry(2.252582954722187) q[5];
rz(3.123067010454827) q[5];
ry(-1.3586485204324343) q[6];
rz(-0.6032596987226454) q[6];
ry(-0.13492135340366573) q[7];
rz(-3.134880353420067) q[7];
ry(2.122785088187497) q[8];
rz(3.08716580361942) q[8];
ry(-0.45727812657502653) q[9];
rz(0.964716217599821) q[9];
ry(-0.9462533806211489) q[10];
rz(-0.2538084080539313) q[10];
ry(-2.500504319488791) q[11];
rz(-2.9741404596156524) q[11];
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
ry(-1.6652212659375767) q[0];
rz(-2.761318687829692) q[0];
ry(-2.028328382372469) q[1];
rz(1.5161533520144914) q[1];
ry(1.3382611317434583) q[2];
rz(0.8174036987422743) q[2];
ry(-2.765215506156717) q[3];
rz(0.11452490864259879) q[3];
ry(-0.25630637690285357) q[4];
rz(-0.3229366073694804) q[4];
ry(1.5698006509062363) q[5];
rz(-0.04364352528137811) q[5];
ry(2.719545609163308) q[6];
rz(-1.5648137968388698) q[6];
ry(1.5776442470963632) q[7];
rz(-1.6234704554707238) q[7];
ry(-1.99429617939163) q[8];
rz(1.868059869940562) q[8];
ry(-2.694391983752022) q[9];
rz(-1.4353155293308237) q[9];
ry(2.240276672724071) q[10];
rz(-1.248422670656638) q[10];
ry(-1.4818391530151729) q[11];
rz(-1.7119442727432277) q[11];
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
ry(-1.0779871663557308) q[0];
rz(2.3782471875327804) q[0];
ry(-0.8417888889966241) q[1];
rz(2.272964318842675) q[1];
ry(-0.24154060278400838) q[2];
rz(-2.273819048063391) q[2];
ry(-1.3498394776585863) q[3];
rz(-0.12522256623574773) q[3];
ry(-0.35557199788623944) q[4];
rz(3.129804210850157) q[4];
ry(-3.1243893030503798) q[5];
rz(-3.096089574527994) q[5];
ry(0.8209574914921988) q[6];
rz(1.629988819887621) q[6];
ry(0.014971161428514096) q[7];
rz(-2.442841226645652) q[7];
ry(1.7662744738622882) q[8];
rz(-2.390172273938196) q[8];
ry(-2.6353194610041495) q[9];
rz(0.8633025038720601) q[9];
ry(2.24087044072213) q[10];
rz(-3.0768834418564697) q[10];
ry(-2.252166341866989) q[11];
rz(-0.8871876797025459) q[11];
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
ry(2.571882725535516) q[0];
rz(-2.410841332319278) q[0];
ry(-1.042008890799754) q[1];
rz(2.4849873532261526) q[1];
ry(2.056397872213492) q[2];
rz(2.2819709706525657) q[2];
ry(-2.500997779617289) q[3];
rz(2.8547143686506526) q[3];
ry(0.58589379992838) q[4];
rz(-0.40525792084839996) q[4];
ry(-0.01886636051145334) q[5];
rz(1.9855311102687951) q[5];
ry(1.0987040686072884) q[6];
rz(0.9931136084562037) q[6];
ry(-3.1263303576569412) q[7];
rz(-2.6098098587631577) q[7];
ry(-0.99744781906991) q[8];
rz(0.6139766370718238) q[8];
ry(-2.9243888647153713) q[9];
rz(1.0465867957820916) q[9];
ry(-2.5058699259522754) q[10];
rz(0.04147823849696853) q[10];
ry(0.7194712804197705) q[11];
rz(-0.505340839004058) q[11];
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
ry(-1.2415510470809554) q[0];
rz(-2.7923756451635606) q[0];
ry(0.43314064608774405) q[1];
rz(-2.5029782904586155) q[1];
ry(0.5085527895393637) q[2];
rz(1.0188833285998236) q[2];
ry(-2.2542290766464923) q[3];
rz(0.9218054356483971) q[3];
ry(1.4651702640975133) q[4];
rz(-2.47674467880262) q[4];
ry(-1.5587000437420508) q[5];
rz(-0.0040399080877754024) q[5];
ry(1.025943153712212) q[6];
rz(1.78001812157603) q[6];
ry(1.5659123819844867) q[7];
rz(3.067784894177582) q[7];
ry(0.5271230705149873) q[8];
rz(-1.9323638716241682) q[8];
ry(2.7238662632821393) q[9];
rz(-1.0834692376628048) q[9];
ry(-0.22827904323113124) q[10];
rz(0.1610997031436151) q[10];
ry(2.0243828633350613) q[11];
rz(-2.9067049382364814) q[11];
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
ry(0.10324854086393703) q[0];
rz(0.180519327675628) q[0];
ry(0.938719109773296) q[1];
rz(0.2150723539710668) q[1];
ry(-1.062099469596947) q[2];
rz(-1.6118827921602412) q[2];
ry(2.1218327393991245) q[3];
rz(-1.270410404191642) q[3];
ry(-2.9378298779986554) q[4];
rz(2.335677266006483) q[4];
ry(-0.19033219752267638) q[5];
rz(1.5740019346367382) q[5];
ry(-2.892704821361619) q[6];
rz(-2.621449209955742) q[6];
ry(-3.0342032671765664) q[7];
rz(-1.637328660463477) q[7];
ry(2.7130619659822304) q[8];
rz(-1.954823109490324) q[8];
ry(-1.916487810275407) q[9];
rz(0.3652167595680807) q[9];
ry(-0.45439391003895846) q[10];
rz(-1.8878796216999612) q[10];
ry(0.09561537517956137) q[11];
rz(-1.8975631486108224) q[11];
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
ry(0.6152535326629627) q[0];
rz(2.738726293447045) q[0];
ry(1.174984078804045) q[1];
rz(0.25759859071163105) q[1];
ry(1.8375788185883664) q[2];
rz(0.46535711339131697) q[2];
ry(-1.8837440319605525) q[3];
rz(2.0560364170327086) q[3];
ry(1.3394131922602774) q[4];
rz(1.538593233733672) q[4];
ry(2.5554579091815546) q[5];
rz(0.10327951341060745) q[5];
ry(-0.9774782304565885) q[6];
rz(-0.6190707404543351) q[6];
ry(2.91698778365799) q[7];
rz(-1.5997557150632586) q[7];
ry(0.9759875104804299) q[8];
rz(0.5862016078593228) q[8];
ry(-1.229615026091488) q[9];
rz(-0.1340061523857617) q[9];
ry(2.3255045871134947) q[10];
rz(1.5567813297290776) q[10];
ry(-1.611087204638829) q[11];
rz(-1.3745412818086291) q[11];
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
ry(2.545340991825786) q[0];
rz(2.707949160651768) q[0];
ry(-0.389666705310316) q[1];
rz(-0.2905372137101153) q[1];
ry(1.2486360249542399) q[2];
rz(-2.0928050954727615) q[2];
ry(2.6103450692326158) q[3];
rz(-1.2587569019934897) q[3];
ry(0.24863146382080445) q[4];
rz(1.3895722666058514) q[4];
ry(3.131207936652126) q[5];
rz(0.11583518850216551) q[5];
ry(1.7787837392996488) q[6];
rz(-2.8664826416632927) q[6];
ry(0.20886926828192642) q[7];
rz(1.6087295899027483) q[7];
ry(-1.8417723060979727) q[8];
rz(1.762587626249017) q[8];
ry(-0.1304126173316982) q[9];
rz(-2.151051931163351) q[9];
ry(-1.2122800971662988) q[10];
rz(0.08720938987473643) q[10];
ry(1.03016200199795) q[11];
rz(2.098317269467392) q[11];
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
ry(1.120111076545199) q[0];
rz(1.5919338128127762) q[0];
ry(-2.0730281238952584) q[1];
rz(-1.4915583101529741) q[1];
ry(-0.5324272629962961) q[2];
rz(-2.7851856488749323) q[2];
ry(-2.6810973346473195) q[3];
rz(-2.655397929526978) q[3];
ry(-1.4596223178614) q[4];
rz(0.799393639712898) q[4];
ry(-2.548434080837984) q[5];
rz(1.5802899667914294) q[5];
ry(-1.6101674284977125) q[6];
rz(-0.5620688213756162) q[6];
ry(-0.5882055486130967) q[7];
rz(1.598689872037598) q[7];
ry(1.8613759427086547) q[8];
rz(-0.8708442504651241) q[8];
ry(1.560589819740709) q[9];
rz(-1.4383092921899079) q[9];
ry(0.37639742227162454) q[10];
rz(0.7739555759691488) q[10];
ry(2.5232065853302053) q[11];
rz(0.4192141107096164) q[11];
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
ry(0.4378335821497961) q[0];
rz(1.2983371712295384) q[0];
ry(1.7471605530942327) q[1];
rz(0.9051522734156116) q[1];
ry(-2.28904198738784) q[2];
rz(0.7909998083464911) q[2];
ry(-2.09908675901405) q[3];
rz(-0.5467047438214845) q[3];
ry(0.33485112140568685) q[4];
rz(1.9728817914553467) q[4];
ry(-1.950952199990245) q[5];
rz(1.5817953927269761) q[5];
ry(1.333623869024687) q[6];
rz(1.2068956087312641) q[6];
ry(3.031031136578513) q[7];
rz(1.5812540647190976) q[7];
ry(0.246634922453981) q[8];
rz(1.6152056228205858) q[8];
ry(-1.2269390447278221) q[9];
rz(2.8181357061814536) q[9];
ry(0.5555638511146553) q[10];
rz(-1.1574165556171874) q[10];
ry(-0.8966679966923596) q[11];
rz(-1.516152302983481) q[11];
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
ry(1.2276515268413755) q[0];
rz(-1.6162179972535846) q[0];
ry(0.2682736016139327) q[1];
rz(1.5115151100303104) q[1];
ry(-2.54866445853788) q[2];
rz(-2.314871723334925) q[2];
ry(1.7789563272972684) q[3];
rz(-0.023364143494921132) q[3];
ry(1.0054983972129916) q[4];
rz(-1.5874478076963796) q[4];
ry(-0.9208204127517439) q[5];
rz(1.5664062418150424) q[5];
ry(-1.2494799136785066) q[6];
rz(2.121472280775259) q[6];
ry(-2.426348459325704) q[7];
rz(-1.5849213444134485) q[7];
ry(-0.6391978208059074) q[8];
rz(-0.8803449710648475) q[8];
ry(2.37698900119653) q[9];
rz(-2.9413936024659777) q[9];
ry(2.0957825183527774) q[10];
rz(2.981898969576923) q[10];
ry(-0.6642018769752633) q[11];
rz(1.4186029575969492) q[11];
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
ry(1.2737809847679546) q[0];
rz(2.326883217045931) q[0];
ry(0.5717559083408) q[1];
rz(2.849675556599451) q[1];
ry(-0.8377039925746016) q[2];
rz(-2.452427967638352) q[2];
ry(0.3948763765793543) q[3];
rz(-0.6821647678629343) q[3];
ry(-2.6381503232175354) q[4];
rz(1.8361710358266246) q[4];
ry(-2.0269836745435343) q[5];
rz(-1.5437492924073206) q[5];
ry(-2.3020255009893864) q[6];
rz(-1.6243803670572279) q[6];
ry(0.7524405719905667) q[7];
rz(-1.7049162239528273) q[7];
ry(0.7155342353511546) q[8];
rz(2.5337086032022023) q[8];
ry(-1.0626323469531758) q[9];
rz(2.359428778832619) q[9];
ry(0.20841038302135007) q[10];
rz(2.6927430024125774) q[10];
ry(-1.9407885196697825) q[11];
rz(2.972634681710446) q[11];
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
ry(1.1195828980663824) q[0];
rz(-1.833681606520152) q[0];
ry(-1.1274613443195607) q[1];
rz(0.05514710018364877) q[1];
ry(0.2108992360230025) q[2];
rz(-2.620778646479462) q[2];
ry(-2.2424087360838856) q[3];
rz(-1.125508230824364) q[3];
ry(-2.4708590079737904) q[4];
rz(-1.9318294852518152) q[4];
ry(-0.20136645704685563) q[5];
rz(-0.2839121514877574) q[5];
ry(1.6115230344838887) q[6];
rz(-2.5911399969719713) q[6];
ry(0.05794247441775192) q[7];
rz(0.8710673692836387) q[7];
ry(1.7130536002055026) q[8];
rz(1.370210307699936) q[8];
ry(2.4254696604136106) q[9];
rz(-0.6762860534813759) q[9];
ry(0.7509866637685249) q[10];
rz(-2.8650032329331045) q[10];
ry(-1.6041827335297125) q[11];
rz(2.0145531428923373) q[11];
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
ry(0.990600095339807) q[0];
rz(2.1048609622145116) q[0];
ry(1.2551629207246133) q[1];
rz(0.8508338450515679) q[1];
ry(-1.2308687802152507) q[2];
rz(-0.03977858640288189) q[2];
ry(-0.5797385768710104) q[3];
rz(2.140761155656997) q[3];
ry(0.7214395994136628) q[4];
rz(-1.3427363770148102) q[4];
ry(-2.5190594800745894) q[5];
rz(1.6008960684108393) q[5];
ry(-0.6046431234691383) q[6];
rz(-0.997836545720986) q[6];
ry(0.8614697810180791) q[7];
rz(0.9448802546657936) q[7];
ry(-0.008896546838024247) q[8];
rz(3.091205017374309) q[8];
ry(1.3332257233436664) q[9];
rz(-1.3511640921041528) q[9];
ry(-1.8527560523197169) q[10];
rz(0.7428326292860225) q[10];
ry(0.5709410137662561) q[11];
rz(-3.1315359853324085) q[11];