//queue()
    //.defer(d3.json, "/my")
    //.await(d3_func);
    //function d3_func(error,myJson){
        d3.csv('/static/2d.csv', function (data) {
            //var data = myJson
            //console.log(data)
        // Variables
          var margin = { top: 50, right: 50, bottom: 50, left: 50 }
          var h = 500 - margin.top - margin.bottom
          var w = 500 - margin.left - margin.right
          var formatPercent = d3.format('.2')
          // Scales
        var colorScale = d3.scale.category20()
        var xScale = d3.scale.linear()
          .domain([
              d3.min([0,d3.min(data,function (d) { return parseFloat(d.x) })]),
              d3.max([0,d3.max(data,function (d) { return parseFloat(d.x) })])
              ])
          .range([0,w])
        var yScale = d3.scale.linear()
          .domain([
              d3.min([0,d3.min(data,function (d) { return parseFloat(d.y) })]),
              d3.max([0,d3.max(data,function (d) { return parseFloat(d.y) })])
              ])
          .range([h,0])
          // SVG
          var svg = d3.select('#graph').append('svg')
              .attr('height',h + margin.top + margin.bottom)
              .attr('width',w + margin.left + margin.right)
            .append('g')
              .attr('transform','translate(' + margin.left + ',' + margin.top + ')')



              svg.append("text")
.attr("x", 130)
.attr("y", -30)
.attr("font-size", 20)
.attr("dy", ".35em")
.style("text-anchor", "start")
.text('Projection Panel');

          // Draw legend

svg.append("rect")
.attr("x",375)
.attr('y', -40)
.attr("width", 18)
.attr("height", 18)
.style("fill", '#9970ab');
svg.append("text")
.attr("x", 400)
.attr("y", -30)
.attr("dy", ".35em")
.style("text-anchor", "start")
.text('Heads');


svg.append("rect").
attr("x", 375)
.attr('y',-20)
.attr("width", 18)
.attr("height", 18)
.style("fill", '#5aae61');
svg.append("text")
.attr("x", 400)
.attr("y", -10)
.attr("dy", ".35em")
.style("text-anchor", "start")
.text('Tails');
              
          // X-axis
          var xAxis = d3.svg.axis()
            .scale(xScale)
            .tickFormat(formatPercent)
            .ticks(5)
            .orient('bottom')
        // Y-axis
          var yAxis = d3.svg.axis()
            .scale(yScale)
            .tickFormat(formatPercent)
            .ticks(5)
            .orient('left')
        // Circles
        var circles = svg.selectAll('circle')
            .data(data)
            .enter()
          .append('circle')
            .attr('cx',function (d) { return xScale(d.x) })
            .attr('cy',function (d) { return yScale(d.y) })
            .attr('r','5')
            .attr('stroke','black')
            .attr('stroke-width',1)
            .attr('fill',function (d,i) { if(d.color==0) return '#9970ab';
                          else return '#5aae61'; })
            .on('mouseover', function () {
              d3.select(this)
                .transition()
                .duration(50)
                .attr('r',8)
                .attr('stroke-width',3)
            })
            .on('mouseout', function () {
              d3.select(this)
                .transition()
                .duration(50)
                .attr('r',5)
                .attr('stroke-width',1)
            })
          .append('title') // Toolt ip
            .text(function (d) { return d.name })
        // X-axis
        svg.append('g')
            .attr('class','axis')
            .attr('transform', 'translate(0,' + h + ')')
            .call(xAxis)
          .append('text') // X-axis Label
            .attr('class','label')
            .attr('y',-10)
            .attr('x',w)
            .attr('dy','.71em')
            .style('text-anchor','end')
            .text('X-Axis')
        // Y-axis
        svg.append('g')
            .attr('class', 'axis')
            .call(yAxis)
          .append('text') // y-axis Label
            .attr('class','label')
            .attr('transform','rotate(-90)')
            .attr('x',0)
            .attr('y',5)
            .attr('dy','.71em')
            .style('text-anchor','end')
            .text('Y-Axis')
      
           var zoom = d3.behavior.zoom()
          .x(xScale)
          .y(yScale)
          .scaleExtent([1, 10])
          .on("zoom", zoomed);
      
      function zoomed() {
        svg.select(".x.axis").call(xAxis);
        svg.select(".y.axis").call(yAxis);
      }
      
      })
      //    }