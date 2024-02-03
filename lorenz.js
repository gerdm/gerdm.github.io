
var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");


// Set canvas size to window size
canvas.width = window.innerWidth;
canvas.height = 200;

// Set line width
ctx.lineWidth = 1.0;
// ctx.strokeStyle = 'black'; // was 'gray' before
var style = 'rgba(0, 0, 0, 1.0)';
ctx.strokeStyle = style;

var s = 10;
var b = 8 / 3;
var r = 28;

var treset = 1000;
var zscale = 4;
var xscale = 7;

function lorenz(x, y, z) {
    // https://en.wikipedia.org/wiki/Lorenz_system
    let dx = s * (y - x);
    let dy = x * (r - z) - y;
    let dz = x * y - b * z;
    return [dx, dy, dz];
}

var vmin = -5;
var vmax = 5;
function random_start() {
    return vmin + (vmax - vmin) * Math.random();
}

function repos_x(x) {
    return x * xscale + canvas.width * 0.5;
}


function repos_z(z) {
    return z * zscale;
}


// Start at the top left (large loss)
var dt = 0.01;
var x = random_start();
var y = random_start();
var z = random_start();

var xpaint = repos_x(x);
var zpaint = repos_z(z);

// Begin the path
ctx.beginPath();
ctx.moveTo(xpaint, zpaint);
ctx.strokeStyle = style;
ctx.stroke();


var i = 0;
setInterval(function () {
    ctx.beginPath();
    ctx.moveTo(xpaint, zpaint);
    let [dx, dy, dz] = lorenz(x, y, z);
    x += dx * dt;
    y += dy * dt;
    z += dz * dt;

    xpaint = repos_x(x);
    zpaint =  repos_z(z);
    ctx.lineTo(xpaint, zpaint);
    ctx.strokeStyle = style;
    ctx.stroke();

    if (i % treset == 0) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    i++;
}, 1);