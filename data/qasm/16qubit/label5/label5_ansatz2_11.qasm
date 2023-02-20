OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.5700578687534632) q[0];
rz(-1.7465860149707926) q[0];
ry(1.5707575156695757) q[1];
rz(-0.0010294454921856756) q[1];
ry(3.1289956335943456) q[2];
rz(-3.0193682424259607) q[2];
ry(-1.5708855541897506) q[3];
rz(-0.0009303318652404968) q[3];
ry(-6.285115131898776e-05) q[4];
rz(-1.0531209544757) q[4];
ry(-0.00018672213689541195) q[5];
rz(1.626401891365119) q[5];
ry(-1.5706814104251463) q[6];
rz(0.0029058116278744) q[6];
ry(1.5708454508147565) q[7];
rz(-0.0011119232663464379) q[7];
ry(-7.283870601160203e-05) q[8];
rz(-3.0503041017312547) q[8];
ry(3.141573288354874) q[9];
rz(1.5085519182006841) q[9];
ry(3.1414737315298833) q[10];
rz(2.8298357069862097) q[10];
ry(3.993821398573516e-05) q[11];
rz(1.6386826815297422) q[11];
ry(1.6840756315694705e-05) q[12];
rz(-2.337662503374475) q[12];
ry(7.476467835942913e-05) q[13];
rz(-0.8306594277615074) q[13];
ry(3.1415603298785832) q[14];
rz(-2.6111804963885517) q[14];
ry(-2.402434804920679e-05) q[15];
rz(-1.0400777320091743) q[15];
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
ry(-3.139937020526344) q[0];
rz(1.5280488762182625) q[0];
ry(0.22518826953875298) q[1];
rz(-1.5698176030282038) q[1];
ry(-3.141451195714655) q[2];
rz(0.11391620796796076) q[2];
ry(-1.0474447012048758) q[3];
rz(-1.028016613743259) q[3];
ry(1.5719732508135564) q[4];
rz(0.00019842312540507834) q[4];
ry(-1.57341606040543) q[5];
rz(-0.2488951923424998) q[5];
ry(-2.582857597188633) q[6];
rz(1.0424951301790086) q[6];
ry(-2.284912235728058) q[7];
rz(0.7678330673709511) q[7];
ry(1.662348509154553) q[8];
rz(-0.5366652212814627) q[8];
ry(-0.0010152091788544482) q[9];
rz(1.8527517510912432) q[9];
ry(-1.5707712675578946) q[10];
rz(1.9096284370588847) q[10];
ry(-1.570456748547697) q[11];
rz(-2.766003909958927) q[11];
ry(-1.0564503670488072) q[12];
rz(0.9759467409655551) q[12];
ry(-1.5672995296070624) q[13];
rz(-0.8265365324378036) q[13];
ry(1.5705967884078738) q[14];
rz(-8.89260896511729e-06) q[14];
ry(-1.5708274023382094) q[15];
rz(6.001677763656233e-05) q[15];
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
ry(-1.7336672778499373) q[0];
rz(-2.240218700327205) q[0];
ry(0.8910341492161757) q[1];
rz(-0.064567445815204) q[1];
ry(-1.5707710892416102) q[2];
rz(-3.081426596869832) q[2];
ry(-0.0005672946968698946) q[3];
rz(1.028948649064972) q[3];
ry(0.9010833323164225) q[4];
rz(-0.005412109658554875) q[4];
ry(3.139751484861025) q[5];
rz(2.8927948086328645) q[5];
ry(-3.1413395520849448) q[6];
rz(0.7960497832458733) q[6];
ry(-3.1412459429745483) q[7];
rz(2.228543650517377) q[7];
ry(3.1414433755764932) q[8];
rz(2.3300326626106873) q[8];
ry(-3.14158916668156) q[9];
rz(0.8686052356578404) q[9];
ry(3.14123165652469) q[10];
rz(-2.8029330758588538) q[10];
ry(-0.00029394336447108316) q[11];
rz(-2.128354887621265) q[11];
ry(3.1414969460614817) q[12];
rz(0.17288037260341882) q[12];
ry(0.0005545457001971566) q[13];
rz(-1.0224459997572521) q[13];
ry(1.1207243037692205) q[14];
rz(0.5736085303900813) q[14];
ry(-1.7869380110365127) q[15];
rz(-1.635527116297193) q[15];
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
ry(-0.06155834514289163) q[0];
rz(0.4387951406769419) q[0];
ry(-3.1415738871403) q[1];
rz(-2.98450629036184) q[1];
ry(1.5706516439511309) q[2];
rz(-1.5758189444947663) q[2];
ry(-2.572547555860314) q[3];
rz(1.573431881987375) q[3];
ry(-1.5718401155328348) q[4];
rz(-2.9560626018446428) q[4];
ry(-1.5704701583248042) q[5];
rz(-1.5191782441429842) q[5];
ry(0.0004321514269869188) q[6];
rz(-1.327293854749346) q[6];
ry(-1.3981222672405975) q[7];
rz(-2.563010998890811) q[7];
ry(3.14140840710501) q[8];
rz(2.10737152641651) q[8];
ry(-0.00015251972197903918) q[9];
rz(-1.7955206065262694) q[9];
ry(1.3969248523986144) q[10];
rz(0.8951804187422541) q[10];
ry(-3.137473701547965) q[11];
rz(-1.752746737796472) q[11];
ry(-0.00017473282020237431) q[12];
rz(0.8030663131967328) q[12];
ry(-0.0030017865340648507) q[13];
rz(1.2660830987732863) q[13];
ry(-3.141102757505422) q[14];
rz(0.5735362814050431) q[14];
ry(3.0855898891315654) q[15];
rz(-1.6355410015928822) q[15];
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
ry(3.1414974689409565) q[0];
rz(0.7815493028855123) q[0];
ry(-3.1410120318240873) q[1];
rz(1.792455076415624) q[1];
ry(-3.1413960479057086) q[2];
rz(0.08012986210720462) q[2];
ry(-0.33685745137056994) q[3];
rz(-0.0021134822307260492) q[3];
ry(-1.571133849201688) q[4];
rz(-3.032838336783512) q[4];
ry(-1.5711742499045775) q[5];
rz(3.139015097801956) q[5];
ry(1.9071836420210586) q[6];
rz(-0.0001514800782952524) q[6];
ry(1.5709384122669166) q[7];
rz(-1.5705789678758324) q[7];
ry(-0.010713188650906499) q[8];
rz(2.391396763579326) q[8];
ry(3.141423436263947) q[9];
rz(2.4249867095022775) q[9];
ry(2.932389394392145) q[10];
rz(-2.04941748525496) q[10];
ry(-1.9809779750377476) q[11];
rz(1.5713717090400365) q[11];
ry(0.29747961022820757) q[12];
rz(1.3289037329807512) q[12];
ry(-3.140913078243918) q[13];
rz(1.2044442277511813) q[13];
ry(1.3327773636297537) q[14];
rz(-1.570792695517837) q[14];
ry(1.268311508989678) q[15];
rz(1.5705702111146937) q[15];
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
ry(3.1235010600827415) q[0];
rz(-1.4512752709409307) q[0];
ry(0.9745468674434848) q[1];
rz(1.500786798761251) q[1];
ry(-0.01988570700090708) q[2];
rz(2.292483721653251) q[2];
ry(-1.5699881615370523) q[3];
rz(-2.8479111928971004) q[3];
ry(-3.103306739950758) q[4];
rz(-2.18424352919217) q[4];
ry(-1.605235411257609) q[5];
rz(1.5267630312884748) q[5];
ry(-1.5708094479696184) q[6];
rz(-0.3572010552038181) q[6];
ry(1.5707747550509572) q[7];
rz(2.6836137512292724) q[7];
ry(-2.7628659575862065) q[8];
rz(-0.07610926229842985) q[8];
ry(1.5705641724404167) q[9];
rz(3.1244249271067135) q[9];
ry(-0.06254689239140099) q[10];
rz(1.639608991055118) q[10];
ry(1.2326881628992412) q[11];
rz(2.2973288206865288) q[11];
ry(-1.9711296152373707) q[12];
rz(2.587379025665889) q[12];
ry(3.125572616595419) q[13];
rz(1.2095914878772742) q[13];
ry(-1.2928842357170272) q[14];
rz(1.5707071435825088) q[14];
ry(1.4849746037101148) q[15];
rz(1.5708419344462228) q[15];
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
ry(-3.141551954822741) q[0];
rz(1.94939683364968) q[0];
ry(-1.3440322592423169e-05) q[1];
rz(1.5345917582976694) q[1];
ry(-7.328839980225156e-05) q[2];
rz(0.5153756910033271) q[2];
ry(-3.141215525801225) q[3];
rz(1.7250260415237433) q[3];
ry(2.3245175732913403) q[4];
rz(1.5957378087534309) q[4];
ry(-0.9047277181501192) q[5];
rz(-0.6452361807924188) q[5];
ry(-0.0001773043495321899) q[6];
rz(-0.7564345401532399) q[6];
ry(-3.1415733748005668) q[7];
rz(-0.42194984971569044) q[7];
ry(1.4330761657776691e-06) q[8];
rz(1.675217032644702) q[8];
ry(0.0019402315025303096) q[9];
rz(1.5869180799444207) q[9];
ry(-3.1386290893649633) q[10];
rz(1.5010535600377481) q[10];
ry(-3.141450284682512) q[11];
rz(1.2677234084845341) q[11];
ry(-3.1412940711494115) q[12];
rz(-3.0168300188765835) q[12];
ry(0.00023597059261104615) q[13];
rz(-0.9935630454054334) q[13];
ry(1.765045654331713) q[14];
rz(1.899227108784805) q[14];
ry(-1.3777697091643697) q[15];
rz(-1.2201786664787797) q[15];
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
ry(-3.141465884888215) q[0];
rz(-0.9319623792993738) q[0];
ry(-8.232036026444435e-05) q[1];
rz(1.5266101881318104) q[1];
ry(3.140723091025581) q[2];
rz(-2.3568696604744717) q[2];
ry(1.5702044554941808) q[3];
rz(2.5670124433644217) q[3];
ry(-0.0021658616148707566) q[4];
rz(-1.7063120880896199) q[4];
ry(-0.0032263481873853195) q[5];
rz(1.9704353866481905) q[5];
ry(1.5708907388436266) q[6];
rz(-1.4854880644228479) q[6];
ry(-0.0007988063886665842) q[7];
rz(1.598980729054653) q[7];
ry(1.5706243521926053) q[8];
rz(0.3004265507637952) q[8];
ry(1.5928764532211386) q[9];
rz(1.569940848088444) q[9];
ry(-0.02724891367294191) q[10];
rz(2.317236090247149) q[10];
ry(0.0006721531035163153) q[11];
rz(0.005433860014388925) q[11];
ry(1.2210892921490455) q[12];
rz(1.5301794894909237) q[12];
ry(2.858741068583129) q[13];
rz(3.1406395206479902) q[13];
ry(-3.14091821016693) q[14];
rz(-3.1325287566198945) q[14];
ry(0.0037205405270630187) q[15];
rz(2.7911200854014235) q[15];
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
ry(-0.0002531200032551695) q[0];
rz(1.4864870274298265) q[0];
ry(3.1414802103648065) q[1];
rz(1.4204784339377934) q[1];
ry(-3.1411610384881796) q[2];
rz(0.8091490948518967) q[2];
ry(-0.00020275600604872137) q[3];
rz(2.1453637853596295) q[3];
ry(-0.0001475325719675532) q[4];
rz(1.595042686207143) q[4];
ry(-6.0700114014375115e-05) q[5];
rz(-1.3014171204406813) q[5];
ry(-3.1415304515910147) q[6];
rz(-1.485434501944552) q[6];
ry(3.141405819175592) q[7];
rz(-1.879717356602927) q[7];
ry(0.00014438381135839506) q[8];
rz(2.8243585838108434) q[8];
ry(1.5748166521265488) q[9];
rz(-2.6739772181968693) q[9];
ry(-1.5742798115287153) q[10];
rz(-1.5692939795619818) q[10];
ry(3.1383963556596064) q[11];
rz(1.7881927059894485) q[11];
ry(-1.543137851926983) q[12];
rz(-2.2128444081186895) q[12];
ry(1.5715817439133621) q[13];
rz(-1.8805451943541651) q[13];
ry(2.131303595188638) q[14];
rz(-2.3746767147021592) q[14];
ry(-1.5703661620552811) q[15];
rz(2.885110483623311) q[15];
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
ry(-0.003498314368052479) q[0];
rz(1.0109019736526006) q[0];
ry(1.5710492722480451) q[1];
rz(1.3594141189779612) q[1];
ry(0.0030651702533323307) q[2];
rz(-1.1267143094074903) q[2];
ry(1.4659087364922625) q[3];
rz(-1.5705719924428534) q[3];
ry(-1.5690354468654955) q[4];
rz(3.1402509436671093) q[4];
ry(1.5708345035700502) q[5];
rz(-0.2639551695639446) q[5];
ry(1.5709499748355746) q[6];
rz(-1.783991274557172) q[6];
ry(3.14133066898296) q[7];
rz(2.622266228524358) q[7];
ry(-1.5707226986952725) q[8];
rz(-3.1415787213704505) q[8];
ry(-6.318636565054344e-05) q[9];
rz(1.103262275133517) q[9];
ry(-1.5710381252180186) q[10];
rz(-0.050578786031074685) q[10];
ry(-3.1388505068038346) q[11];
rz(-0.3290427563874855) q[11];
ry(3.1415396884584865) q[12];
rz(0.5765066339538389) q[12];
ry(3.141264050451348) q[13];
rz(-2.840367298823522) q[13];
ry(0.01125757642072854) q[14];
rz(0.5668930650646035) q[14];
ry(-3.141509822506303) q[15];
rz(-1.352013862025756) q[15];
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
ry(-3.1406880571564866) q[0];
rz(0.9081746352037069) q[0];
ry(-0.00037531769847998887) q[1];
rz(1.7819370780276171) q[1];
ry(0.0006423779799291898) q[2];
rz(-2.185133014173351) q[2];
ry(-1.570983328044675) q[3];
rz(-0.8283731430551868) q[3];
ry(1.5708980263848966) q[4];
rz(-1.4237777480841005) q[4];
ry(3.1415848559036315) q[5];
rz(1.6172983070866387) q[5];
ry(1.1874803742095708e-05) q[6];
rz(-1.4623902165615923) q[6];
ry(-0.0003114247540931316) q[7];
rz(1.7169573599484416) q[7];
ry(1.5703601261608346) q[8];
rz(0.8629530854134009) q[8];
ry(1.5707646530575274) q[9];
rz(2.530267161608868) q[9];
ry(-1.5707192902207552) q[10];
rz(-2.3820928547479405) q[10];
ry(1.570553196278638) q[11];
rz(1.367402332635508) q[11];
ry(3.1407241005691127) q[12];
rz(0.4105306678640819) q[12];
ry(-3.70122069555472e-05) q[13];
rz(0.6462481833918564) q[13];
ry(-3.141474826432866) q[14];
rz(-1.2578440026594715) q[14];
ry(-3.1415575267257903) q[15];
rz(-1.2545868736920234) q[15];
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
ry(3.1399352667153484) q[0];
rz(-0.2670501226977999) q[0];
ry(-1.5706642900047258) q[1];
rz(-5.9740708919110786e-05) q[1];
ry(0.0016426038458171896) q[2];
rz(-2.747781428984149) q[2];
ry(3.1415670012642045) q[3];
rz(-2.3991293865092462) q[3];
ry(3.80535931672199e-05) q[4];
rz(0.1320414708707978) q[4];
ry(3.141556059793766) q[5];
rz(-1.8261389172420213) q[5];
ry(-1.570829282807324) q[6];
rz(-4.260802750710724e-05) q[6];
ry(1.5708069959187991) q[7];
rz(3.0920831838637697) q[7];
ry(-3.1415745333434626) q[8];
rz(-2.278634186622375) q[8];
ry(-6.975730478703612e-05) q[9];
rz(0.6110897433033451) q[9];
ry(6.557640380398059e-06) q[10];
rz(2.382063509102011) q[10];
ry(2.4145260129543357e-05) q[11];
rz(1.7738378860246453) q[11];
ry(-2.2687350551997557e-05) q[12];
rz(-0.7356308466563757) q[12];
ry(3.1415417255569777) q[13];
rz(-2.7347442109705993) q[13];
ry(-3.413666356050976e-05) q[14];
rz(-2.983681193793577) q[14];
ry(7.791738968504698e-06) q[15];
rz(-1.544703117559008) q[15];
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
ry(1.5708410130854242) q[0];
rz(0.40394159624131104) q[0];
ry(-1.5708264851962062) q[1];
rz(1.298004851463511) q[1];
ry(1.5708386501520346) q[2];
rz(-1.3107095941819074) q[2];
ry(-1.5708325788067612) q[3];
rz(-1.233948731071736) q[3];
ry(3.1415784019368567) q[4];
rz(-1.9655922921450726) q[4];
ry(-8.731667518802055e-06) q[5];
rz(2.0848799929598023) q[5];
ry(-1.571126628389139) q[6];
rz(-3.1414930651029964) q[6];
ry(-2.625529710531538e-05) q[7];
rz(-3.091987669277781) q[7];
ry(-1.5708382812620703) q[8];
rz(-2.625356141214869) q[8];
ry(1.5706876637547915) q[9];
rz(3.1242774419905364) q[9];
ry(-1.5707978914032346) q[10];
rz(3.1415140474385312) q[10];
ry(1.567360384010982) q[11];
rz(-3.1250492667158927) q[11];
ry(-3.868862825842546e-05) q[12];
rz(-1.5979154737217547) q[12];
ry(3.1415089374688447) q[13];
rz(-2.1283274068452362) q[13];
ry(-3.1415782716236995) q[14];
rz(3.1250237754385797) q[14];
ry(4.543411702664694e-05) q[15];
rz(0.19481252989106754) q[15];
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
ry(-3.1415808291299023) q[0];
rz(2.2417775885715976) q[0];
ry(-1.7757540116648355e-06) q[1];
rz(0.9280550417632654) q[1];
ry(3.141581406071607) q[2];
rz(-0.07121960965830798) q[2];
ry(-3.141574960344877) q[3];
rz(1.3073127499449466) q[3];
ry(3.14151651503514) q[4];
rz(-2.247482085309943) q[4];
ry(3.141503616336249) q[5];
rz(3.0916493902521287) q[5];
ry(-1.5708651701092693) q[6];
rz(-0.9330676792498358) q[6];
ry(1.5708464482406512) q[7];
rz(1.5704112452861985) q[7];
ry(0.00017988642346722372) q[8];
rz(-1.1803573495020405) q[8];
ry(-3.141283729185726) q[9];
rz(3.124204051686739) q[9];
ry(-1.5709114084720914) q[10];
rz(-3.0900488166857243) q[10];
ry(3.141133892393198) q[11];
rz(-3.1252246698906303) q[11];
ry(1.5745923534678672) q[12];
rz(-3.138925122404226) q[12];
ry(6.760038032844677e-05) q[13];
rz(1.1327134621928572) q[13];
ry(-0.003444239261935803) q[14];
rz(-2.9737542790667453) q[14];
ry(0.00013527064314265686) q[15];
rz(0.15100671747174177) q[15];
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
ry(-0.7432690882425613) q[0];
rz(0.6081946866952495) q[0];
ry(-3.1415557813137633) q[1];
rz(0.5744408070672691) q[1];
ry(0.0002847185365499557) q[2];
rz(-0.3649887416484337) q[2];
ry(-0.00012163804175351203) q[3];
rz(2.090187856811279) q[3];
ry(-1.5708789315926346) q[4];
rz(2.1715916476276624) q[4];
ry(-1.5701885572242453) q[5];
rz(-1.6304715846259956) q[5];
ry(7.341380744748161e-05) q[6];
rz(-2.0037509962033626) q[6];
ry(-1.570890841446677) q[7];
rz(1.506202372071692) q[7];
ry(0.00018242648037786335) q[8];
rz(2.8401669373333) q[8];
ry(1.5704950660188055) q[9];
rz(1.4915135106931379) q[9];
ry(-3.0285949862235038) q[10];
rz(0.6565942634426332) q[10];
ry(-1.5717076425284213) q[11];
rz(1.4965744190889594) q[11];
ry(-1.5708477618139416) q[12];
rz(0.6009330585063839) q[12];
ry(-0.000904742492690895) q[13];
rz(-2.313089329912359) q[13];
ry(1.428507240896009) q[14];
rz(0.6007940270675421) q[14];
ry(-0.0008269077123612688) q[15];
rz(-1.5260918786251376) q[15];