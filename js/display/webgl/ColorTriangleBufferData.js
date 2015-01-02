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
    createFromPath: function( path, z ) {
      assert && assert( z !== undefined );

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
      for ( var k = 0; k < triangles.length; k++ ) {
        var triangle = triangles[k];
        for ( var zz = 0; zz < triangle.points_.length; zz++ ) {
          var pt = triangle.points_[zz];

          // Mutate the vertices a bit to see what is going on.  Or not.
          var randFactor = 0;
          this.vertexArray.push( pt.x + Math.random() * randFactor, pt.y + Math.random() * randFactor, z );
          this.colors.push( color.red / 255, color.green / 255, color.blue / 255, color.alpha );
        }
      }
    },
    createFromTriangle: function( x1, y1, x2, y2, x3, y3, color, z ) {
      assert && assert( z !== undefined );

      color = new Color( color );
      var r = color.red / 255;
      var g = color.green / 255;
      var b = color.blue / 255;
      var a = color.alpha;

      var colorTriangleBufferData = this;
      var index = this.vertexArray.length;
      colorTriangleBufferData.vertexArray.push(
        // Top left
        x1, y1, z,
        x2, y2, z,
        x3, y3, z
      );

      // Add the same color for all vertices (solid fill rectangle).
      // TODO: some way to reduce this amount of elements!
      colorTriangleBufferData.colors.push(
        r, g, b, a,
        r, g, b, a,
        r, g, b, a
      );

      //Track the index so it can delete itself, update itself, etc.
      //TODO: Move to a separate class.
      return {
        index: index,
        endIndex: colorTriangleBufferData.vertexArray.length,
        setTriangle: function( x1, y1, x2, y2, x3, y3 ) {
          colorTriangleBufferData.vertexArray[index + 0] = x1;
          colorTriangleBufferData.vertexArray[index + 1] = y1;
          colorTriangleBufferData.vertexArray[index + 3] = x2;
          colorTriangleBufferData.vertexArray[index + 4] = y2;
          colorTriangleBufferData.vertexArray[index + 6] = x3;
          colorTriangleBufferData.vertexArray[index + 7] = y3;
        },
        setZ: function( z ) {
          colorTriangleBufferData.vertexArray[index + 2] = z;
          colorTriangleBufferData.vertexArray[index + 5] = z;
          colorTriangleBufferData.vertexArray[index + 8] = z;
        }
      };
    },
    createFromRectangle: function( rectangle, z ) {
      assert && assert( z !== undefined );

      var color = new Color( rectangle.fill );
      return this.createRectangle( rectangle.rectX, rectangle.rectY, rectangle.rectWidth, rectangle.rectHeight, color.red / 255, color.green / 255, color.blue / 255, color.alpha, z );
    },
    createRectangle: function( x, y, width, height, r, g, b, a, z ) {
      assert && assert( z !== undefined );

      var colorTriangleBufferData = this;
      var index = this.vertexArray.length;
      this.vertexArray.push(
        // Top left
        x, y, z,
        (x + width), y, z,
        x, y + height, z,

        // Bottom right
        (x + width), y + height, z,
        (x + width), y, z,
        x, y + height, z
      );

      // Add the same color for all vertices (solid fill rectangle).
      // TODO: some way to reduce this amount of elements!
      this.colors.push(
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
        endIndex: colorTriangleBufferData.vertexArray.length,
        setXWidth: function( x, width ) {
          colorTriangleBufferData.vertexArray[index + 0] = x;
          colorTriangleBufferData.vertexArray[index + 3] = x + width;
          colorTriangleBufferData.vertexArray[index + 6] = x;
          colorTriangleBufferData.vertexArray[index + 9] = x + width;
          colorTriangleBufferData.vertexArray[index + 12] = x + width;
          colorTriangleBufferData.vertexArray[index + 15] = x;
        },
        setRect: function( x, y, width, height ) {

          colorTriangleBufferData.vertexArray[index + 0] = x;
          colorTriangleBufferData.vertexArray[index + 1] = y;

          colorTriangleBufferData.vertexArray[index + 3] = x + width;
          colorTriangleBufferData.vertexArray[index + 4] = y;

          colorTriangleBufferData.vertexArray[index + 6] = x;
          colorTriangleBufferData.vertexArray[index + 7] = y + height;

          colorTriangleBufferData.vertexArray[index + 9] = x + width;
          colorTriangleBufferData.vertexArray[index + 10] = y + height;

          colorTriangleBufferData.vertexArray[index + 12] = x + width;
          colorTriangleBufferData.vertexArray[index + 13] = y;

          colorTriangleBufferData.vertexArray[index + 15] = x;
          colorTriangleBufferData.vertexArray[index + 16] = y + height;
        }
      };
    },

    createStar: function( _x, _y, _innerRadius, _outerRadius, _totalAngle, r, g, b, a, z ) {
      assert && assert( z !== undefined );

      var colorTriangleBufferData = this;
      var index = this.vertexArray.length;
      for ( var i = 0; i < 18; i++ ) {
        this.vertexArray.push( 0 );
      }

      // Add the same color for all vertices (solid fill star).
      // TODO: some way to reduce this amount of elements!
      this.colors.push(
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
          colorTriangleBufferData.vertexArray[index + 0] = points[0].x;
          colorTriangleBufferData.vertexArray[index + 1] = points[0].y;
          colorTriangleBufferData.vertexArray[index + 2] = points[3].x;
          colorTriangleBufferData.vertexArray[index + 3] = points[3].y;
          colorTriangleBufferData.vertexArray[index + 4] = points[6].x;
          colorTriangleBufferData.vertexArray[index + 5] = points[6].y;

          colorTriangleBufferData.vertexArray[index + 6] = points[8].x;
          colorTriangleBufferData.vertexArray[index + 7] = points[8].y;
          colorTriangleBufferData.vertexArray[index + 8] = points[2].x;
          colorTriangleBufferData.vertexArray[index + 9] = points[2].y;
          colorTriangleBufferData.vertexArray[index + 10] = points[5].x;
          colorTriangleBufferData.vertexArray[index + 11] = points[5].y;

          colorTriangleBufferData.vertexArray[index + 12] = points[0].x;
          colorTriangleBufferData.vertexArray[index + 13] = points[0].y;
          colorTriangleBufferData.vertexArray[index + 14] = points[7].x;
          colorTriangleBufferData.vertexArray[index + 15] = points[7].y;
          colorTriangleBufferData.vertexArray[index + 16] = points[4].x;
          colorTriangleBufferData.vertexArray[index + 17] = points[4].y;
        }
      };
      myStar.setStar( _x, _y, _innerRadius, _outerRadius, _totalAngle );
      return myStar;
    }
  } );
} );