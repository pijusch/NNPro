// Set the dimensions of the canvas / graph
var margin = {top: 30, right: 20, bottom: 30, left: 50},
    width = 300 - margin.left - margin.right,
    height = 200 - margin.top - margin.bottom;

// Parse the date / time

// Set the ranges
var x = d3.scale.linear().range([0, width]);
var y = d3.scale.linear().range([height, 0]);

// Define the axes
var xAxis = d3.svg.axis().scale(x)
    .orient("bottom").ticks(5);

var yAxis = d3.svg.axis().scale(y)
    .orient("left").ticks(5);

// Define the line
var valueline = d3.svg.line()
    .x(function(d) { return x(parseFloat(d.iter)); })
    .y(function(d) { return y(parseFloat(d.loss)); });
    
// Adds the svg canvas
var svg = d3.select("#loss")
    .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform", 
              "translate(" + margin.left + "," + margin.top + ")");


              svg.append("text")
.attr("x", 50)
.attr("y", -10)
.attr("font-size", 20)
.attr("dy", ".35em")
.style("text-anchor", "start")
.text('Loss Graph');


// Get the data
d3.csv("/static/loss.csv", function(error, data) {
    console.log(data)
/*data.forEach(function(d) {
        d.iter = d.iter;
        d.loss = +d.loss;
    });*/

    // Scale the range of the data
    x.domain([d3.min(data, function(d) { return parseFloat(d.iter); }), d3.max(data, function(d) { return parseFloat(d.iter); })]);
    y.domain([0, d3.max(data, function(d) { return parseFloat(d.loss); })]);

    svg.selectAll('circle')
            .data(data)
            .enter()
          .append('circle')
    .attr('cx',function (d) { return x(d.iter) })
    .attr('cy',function (d) { return y(d.loss) })
    .attr('r','2')
    .attr('stroke','black');

    // Add the valueline path.
    svg.append("path")
        .attr("class", "line")
        .attr("d", valueline(data));

    // Add the X Axis
    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    // Add the Y Axis
    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis);

});