//  Copyright 2002-2014, University of Colorado Boulder

/**
 * This WebGL renderer is used to draw colored triangles.  Vertices are allocated for geometry + colors, and can be updated
 * dynamically.
 * TODO: Can this same pattern be used for interleaved texture coordinates? (Or other interleaved data?)
 * TODO: Work in progress, much to be done here!
 * TODO: Add this file to the list of scenery files (for jshint, etc.)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Color = require( 'SCENERY/util/Color' );

  /**
   * TODO - Merge all these individual array into a single structure to improve performance
   * @constructor
   */
  function ColorTriangleBufferData() {

    //TODO: Use Float32Array -- though we will have to account for the fact that they have a fixed size
    this.vertexArray = [];
    this.colors = [];
  }

  return inherit( Object, ColorTriangleBufferData, {

    /**
     * Add geometry and color for a scenery path using sampling + triangulation.
     * Uses poly2tri for triangulation
     * @param path
     */
    createFromPath: function( path ) {
      var shape = path.shape;
      var color = new Color( path.fill );
      var linear = shape.toPiecewiseLinear( {} );
      var subpaths = linear.subpaths;

      // Output to a string for ease of debugging within http://r3mi.github.io/poly2tri.js/
      var string = '';

      // Output the contour to an array of poly2tri.Point
      var contour = [];

      for ( var i = 0; i < subpaths.length; i++ ) {
        var subpath = subpaths[i];
        for ( var k = 0; k < subpath.points.length; k++ ) {

          string = string + '' + subpath.points[k].x + ' ' + subpath.points[k].y + '\n';

          //Add the points into the contour, but don't duplicate the last point.
          //TODO: how to handle closed vs open shapes
          if ( k < subpath.points.length - 1 ) {
            contour.push( new poly2tri.Point( subpath.points[k].x, subpath.points[k].y ) );
          }
        }
      }

      // Triangulate using poly2tri
      // Circle linearization is creating some duplicated points, so bail on those for now.
      var triangles;
      try {
        triangles = new poly2tri.SweepContext( contour ).triangulate().getTriangles();
      }
      catch( error ) {
        console.log( 'error in triangulation', error );
        triangles = [];
      }

      // Add the triangulated geometry into the array buffer.
      for ( var z = 0; z < triangles.length; z++ ) {
        var triangle = triangles[z];
        for ( var zz = 0; zz < triangle.points_.length; zz++ ) {
          var pt = triangle.points_[zz];

          // Mutate the vertices a bit to see what is going on.  Or not.
          var randFactor = 0;
          this.vertexArray.push( pt.x + Math.random() * randFactor, pt.y + Math.random() * randFactor );
          this.colors.push( color.red / 255, color.green / 255, color.blue / 255, color.alpha );
        }
      }
    },
    createFromTriangle: function( x1, y1, x2, y2, x3, y3, color, depth ) {

      color = new Color( color );
      var r = color.red / 255;
      var g = color.green / 255;
      var b = color.blue / 255;
      var a = color.alpha;

      var ColorTriangleBufferData = this;
      var index = this.vertexArray.length;
      ColorTriangleBufferData.vertexArray.push(
        // Top left
        x1, y1, depth,
        x2, y2, depth,
        x3, y3, depth
      );

      // Add the same color for all vertices (solid fill rectangle).
      // TODO: some way to reduce this amount of elements!
      ColorTriangleBufferData.colors.push(
        r, g, b, a,
        r, g, b, a,
        r, g, b, a
      );

      //Track the index so it can delete itself, update itself, etc.
      //TODO: Move to a separate class.
      return {
        index: index,
        endIndex: ColorTriangleBufferData.vertexArray.length,
        setTriangle: function( x1, y1, x2, y2, x3, y3 ) {
          ColorTriangleBufferData.vertexArray[index + 0] = x1;
          ColorTriangleBufferData.vertexArray[index + 1] = y1;
          ColorTriangleBufferData.vertexArray[index + 2] = x2;
          ColorTriangleBufferData.vertexArray[index + 3] = y2;
          ColorTriangleBufferData.vertexArray[index + 4] = x3;
          ColorTriangleBufferData.vertexArray[index + 5] = y3;
        }
      };
    },
    createFromRectangle: function( rectangle, depth ) {
      var color = new Color( rectangle.fill );
      return this.createRectangle( rectangle.rectX, rectangle.rectY, rectangle.rectWidth, rectangle.rectHeight, color.red / 255, color.green / 255, color.blue / 255, color.alpha, depth );
    },
    createRectangle: function( x, y, width, height, r, g, b, a, depth ) {
      var ColorTriangleBufferData = this;
      var index = this.vertexArray.length;
      ColorTriangleBufferData.vertexArray.push(
        // Top left
        x, y, depth,
        (x + width), y, depth,
        x, y + height, depth,

        // Bottom right
        (x + width), y + height, depth,
        (x + width), y, depth,
        x, y + height, depth
      );

      // Add the same color for all vertices (solid fill rectangle).
      // TODO: some way to reduce this amount of elements!
      ColorTriangleBufferData.colors.push(
        r, g, b, a,
        r, g, b, a,
        r, g, b, a,
        r, g, b, a,
        r, g, b, a,
        r, g, b, a
      );

      //Track the index so it can delete itself, update itself, etc.
      //TODO: Move to a separate class.
      return {
        initialState: {x: x, y: y, width: width, height: height},
        index: index,
        endIndex: ColorTriangleBufferData.vertexArray.length,
        setXWidth: function( x, width ) {
          ColorTriangleBufferData.vertexArray[index] = x;
          ColorTriangleBufferData.vertexArray[index + 2] = x + width;
          ColorTriangleBufferData.vertexArray[index + 4] = x;
          ColorTriangleBufferData.vertexArray[index + 6] = x + width;
          ColorTriangleBufferData.vertexArray[index + 8] = x + width;
          ColorTriangleBufferData.vertexArray[index + 10] = x;
        },
        setRect: function( x, y, width, height ) {

          ColorTriangleBufferData.vertexArray[index] = x;
          ColorTriangleBufferData.vertexArray[index + 1] = y;

          ColorTriangleBufferData.vertexArray[index + 2] = x + width;
          ColorTriangleBufferData.vertexArray[index + 3] = y;

          ColorTriangleBufferData.vertexArray[index + 4] = x;
          ColorTriangleBufferData.vertexArray[index + 5] = y + height;

          ColorTriangleBufferData.vertexArray[index + 6] = x + width;
          ColorTriangleBufferData.vertexArray[index + 7] = y + height;

          ColorTriangleBufferData.vertexArray[index + 8] = x + width;
          ColorTriangleBufferData.vertexArray[index + 9] = y;

          ColorTriangleBufferData.vertexArray[index + 10] = x;
          ColorTriangleBufferData.vertexArray[index + 11] = y + height;
        }
      };
    },

    createStar: function( _x, _y, _innerRadius, _outerRadius, _totalAngle, r, g, b, a ) {
      var ColorTriangleBufferData = this;
      var index = this.vertexArray.length;
      for ( var i = 0; i < 18; i++ ) {
        ColorTriangleBufferData.vertexArray.push( 0 );
      }

      // Add the same color for all vertices (solid fill star).
      // TODO: some way to reduce this amount of elements!
      ColorTriangleBufferData.colors.push(
        r, g, b, a,
        r, g, b, a,
        r, g, b, a,

        r, g, b, a,
        r, g, b, a,
        r, g, b, a,

        r, g, b, a,
        r, g, b, a,
        r, g, b, a
      );

      //Track the index so it can delete itself, update itself, etc.
      var myStar = {
        initialState: {_x: _x, _y: _y, _innerRadius: _innerRadius, _outerRadius: _outerRadius, _totalAngle: _totalAngle},
        index: index,
        setStar: function( _x, _y, _innerRadius, _outerRadius, _totalAngle ) {

          var points = [];
          //Create the points for a filled-in star, which will be used to compute the geometry of a partial star.
          for ( i = 0; i < 10; i++ ) {

            //Start at the top and proceed clockwise
            var angle = i / 10 * Math.PI * 2 - Math.PI / 2 + _totalAngle;
            var radius = i % 2 === 0 ? _outerRadius : _innerRadius;
            var x = radius * Math.cos( angle ) + _x;
            var y = radius * Math.sin( angle ) + _y;
            points.push( {x: x, y: y} );
          }

          var index = this.index;
          ColorTriangleBufferData.vertexArray[index + 0] = points[0].x;
          ColorTriangleBufferData.vertexArray[index + 1] = points[0].y;
          ColorTriangleBufferData.vertexArray[index + 2] = points[3].x;
          ColorTriangleBufferData.vertexArray[index + 3] = points[3].y;
          ColorTriangleBufferData.vertexArray[index + 4] = points[6].x;
          ColorTriangleBufferData.vertexArray[index + 5] = points[6].y;

          ColorTriangleBufferData.vertexArray[index + 6] = points[8].x;
          ColorTriangleBufferData.vertexArray[index + 7] = points[8].y;
          ColorTriangleBufferData.vertexArray[index + 8] = points[2].x;
          ColorTriangleBufferData.vertexArray[index + 9] = points[2].y;
          ColorTriangleBufferData.vertexArray[index + 10] = points[5].x;
          ColorTriangleBufferData.vertexArray[index + 11] = points[5].y;

          ColorTriangleBufferData.vertexArray[index + 12] = points[0].x;
          ColorTriangleBufferData.vertexArray[index + 13] = points[0].y;
          ColorTriangleBufferData.vertexArray[index + 14] = points[7].x;
          ColorTriangleBufferData.vertexArray[index + 15] = points[7].y;
          ColorTriangleBufferData.vertexArray[index + 16] = points[4].x;
          ColorTriangleBufferData.vertexArray[index + 17] = points[4].y;
        }
      };
      myStar.setStar( _x, _y, _innerRadius, _outerRadius, _totalAngle );
      return myStar;
    }
  } );
} );