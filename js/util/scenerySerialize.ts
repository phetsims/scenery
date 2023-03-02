// Copyright 2021-2023, University of Colorado Boulder

/**
 * Serializes a generalized object
 * @deprecated
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import ReadOnlyProperty from '../../../axon/js/ReadOnlyProperty.js';
import inheritance from '../../../phet-core/js/inheritance.js';
import { CanvasContextWrapper, CanvasNode, Circle, Color, Display, DOM, Gradient, Image, Line, LinearGradient, Node, Paint, PAINTABLE_DEFAULT_OPTIONS, Path, Pattern, RadialGradient, Rectangle, scenery, Text, WebGLNode } from '../imports.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';

const scenerySerialize = ( value: unknown ): IntentionalAny => {
  if ( value instanceof Vector2 ) {
    return {
      type: 'Vector2',
      x: ( value ).x,
      y: ( value ).y
    };
  }
  else if ( value instanceof Matrix3 ) {
    return {
      type: 'Matrix3',
      m00: value.m00(),
      m01: value.m01(),
      m02: value.m02(),
      m10: value.m10(),
      m11: value.m11(),
      m12: value.m12(),
      m20: value.m20(),
      m21: value.m21(),
      m22: value.m22()
    };
  }
  else if ( value instanceof Bounds2 ) {
    const bounds = value;
    return {
      type: 'Bounds2',
      maxX: bounds.maxX,
      maxY: bounds.maxY,
      minX: bounds.minX,
      minY: bounds.minY
    };
  }
  else if ( value instanceof Shape ) {
    return {
      type: 'Shape',
      path: value.getSVGPath()
    };
  }
  else if ( Array.isArray( value ) ) {
    return {
      type: 'Array',
      value: value.map( scenerySerialize )
    };
  }
  else if ( value instanceof Color ) {
    return {
      type: 'Color',
      red: value.red,
      green: value.green,
      blue: value.blue,
      alpha: value.alpha
    };
  }
  else if ( value instanceof ReadOnlyProperty ) {
    return {
      type: 'Property',
      value: scenerySerialize( value.value )
    };
  }
  else if ( Paint && value instanceof Paint ) {
    const paintSerialization: IntentionalAny = {};

    if ( value.transformMatrix ) {
      paintSerialization.transformMatrix = scenerySerialize( value.transformMatrix );
    }

    if ( Gradient && ( value instanceof RadialGradient || value instanceof LinearGradient ) ) {
      paintSerialization.stops = value.stops.map( stop => {
        return {
          ratio: stop.ratio,
          stop: scenerySerialize( stop.color )
        };
      } );

      paintSerialization.start = scenerySerialize( value.start );
      paintSerialization.end = scenerySerialize( value.end );

      if ( LinearGradient && value instanceof LinearGradient ) {
        paintSerialization.type = 'LinearGradient';
      }
      else if ( RadialGradient && value instanceof RadialGradient ) {
        paintSerialization.type = 'RadialGradient';
        paintSerialization.startRadius = value.startRadius;
        paintSerialization.endRadius = value.endRadius;
      }
    }

    if ( Pattern && value instanceof Pattern ) {
      paintSerialization.type = 'Pattern';
      paintSerialization.url = value.image.src;
    }

    return paintSerialization;
  }
  else if ( value instanceof Node ) {
    const node = value;

    const options: IntentionalAny = {};
    const setup: IntentionalAny = {
      // maxWidth
      // maxHeight
      // clipArea
      // mouseArea
      // touchArea
      // matrix
      // localBounds
      // children {Array.<number>} - IDs
      // hasInputListeners {boolean}

    };

    [
      'visible',
      'opacity',
      'disabledOpacity',
      'pickable',
      'inputEnabled',
      'cursor',
      'transformBounds',
      'renderer',
      'usesOpacity',
      'layerSplit',
      'cssTransform',
      'excludeInvisible',
      'webglScale',
      'preventFit'
    ].forEach( simpleKey => {
      // @ts-expect-error
      if ( node[ simpleKey ] !== Node.DEFAULT_NODE_OPTIONS[ simpleKey ] ) {
        // @ts-expect-error
        options[ simpleKey ] = node[ simpleKey ];
      }
    } );


    // From ParallelDOM
    [
      'tagName',
      'innerContent',
      'accessibleName',
      'helpText'
    ].forEach( simpleKey => {

      // All default to null
      // @ts-expect-error
      if ( node[ simpleKey ] !== null ) {
        // @ts-expect-error
        options[ simpleKey ] = node[ simpleKey ];
      }
    } );

    [
      'maxWidth',
      'maxHeight',
      'clipArea',
      'mouseArea',
      'touchArea'
    ].forEach( serializedKey => {
      // @ts-expect-error
      if ( node[ serializedKey ] !== Node.DEFAULT_NODE_OPTIONS[ serializedKey ] ) {
        // @ts-expect-error
        setup[ serializedKey ] = scenerySerialize( node[ serializedKey ] );
      }
    } );
    if ( !node.matrix.isIdentity() ) {
      setup.matrix = scenerySerialize( node.matrix );
    }
    if ( node._localBoundsOverridden ) {
      setup.localBounds = scenerySerialize( node.localBounds );
    }
    setup.children = node.children.map( child => {
      return child.id;
    } );
    setup.hasInputListeners = node.inputListeners.length > 0;

    const serialization: IntentionalAny = {
      id: node.id,
      type: 'Node',
      types: inheritance( node.constructor ).map( type => type.name ).filter( name => {
        return name && name !== 'Object' && name !== 'Node';
      } ),
      name: node.constructor.name,
      options: options,
      setup: setup
    };

    if ( Path && node instanceof Path ) {
      serialization.type = 'Path';
      setup.path = scenerySerialize( node.shape );
      if ( node.boundsMethod !== Path.DEFAULT_PATH_OPTIONS.boundsMethod ) {
        options.boundsMethod = node.boundsMethod;
      }
    }

    if ( Circle && node instanceof Circle ) {
      serialization.type = 'Circle';
      options.radius = node.radius;
    }

    if ( Line && node instanceof Line ) {
      serialization.type = 'Line';
      options.x1 = node.x1;
      options.y1 = node.y1;
      options.x2 = node.x2;
      options.y2 = node.y2;
    }

    if ( Rectangle && node instanceof Rectangle ) {
      serialization.type = 'Rectangle';
      options.rectX = node.rectX;
      options.rectY = node.rectY;
      options.rectWidth = node.rectWidth;
      options.rectHeight = node.rectHeight;
      options.cornerXRadius = node.cornerXRadius;
      options.cornerYRadius = node.cornerYRadius;
    }

    if ( Text && node instanceof Text ) {
      serialization.type = 'Text';
      // TODO: defaults for Text?
      if ( node.boundsMethod !== 'hybrid' ) {
        options.boundsMethod = node.boundsMethod;
      }
      options.string = node.string;
      options.font = node.font;
    }

    if ( Image && node instanceof Image ) {
      serialization.type = 'Image';
      [
        'imageOpacity',
        'initialWidth',
        'initialHeight',
        'mipmapBias',
        'mipmapInitialLevel',
        'mipmapMaxLevel'
      ].forEach( simpleKey => {
        // @ts-expect-error
        if ( node[ simpleKey ] !== Image.DEFAULT_IMAGE_OPTIONS[ simpleKey ] ) {
          // @ts-expect-error
          options[ simpleKey ] = node[ simpleKey ];
        }
      } );

      setup.width = node.imageWidth;
      setup.height = node.imageHeight;

      // Initialized with a mipmap
      if ( node._mipmapData ) {
        setup.imageType = 'mipmapData';
        setup.mipmapData = node._mipmapData.map( level => {
          return {
            url: level.url,
            width: level.width,
            height: level.height
            // will reconstitute img {HTMLImageElement} and canvas {HTMLCanvasElement}
          };
        } );
      }
      else {
        if ( node._mipmap ) {
          setup.generateMipmaps = true;
        }
        if ( node._image instanceof HTMLImageElement ) {
          setup.imageType = 'image';
          setup.src = node._image.src;
        }
        else if ( node._image instanceof HTMLCanvasElement ) {
          setup.imageType = 'canvas';
          setup.src = node._image.toDataURL();
        }
      }
    }

    if ( ( CanvasNode && node instanceof CanvasNode ) ||
         ( WebGLNode && node instanceof WebGLNode ) ) {
      serialization.type = ( CanvasNode && node instanceof CanvasNode ) ? 'CanvasNode' : 'WebGLNode';

      setup.canvasBounds = scenerySerialize( node.canvasBounds );

      // Identify the approximate scale of the node
      // let scale = Math.min( 5, node._drawables.length ? ( 1 / _.mean( node._drawables.map( drawable => {
      //   const scaleVector = drawable.instance.trail.getMatrix().getScaleVector();
      //   return ( scaleVector.x + scaleVector.y ) / 2;
      // } ) ) ) : 1 );
      const scale = 1;
      const canvas = document.createElement( 'canvas' );
      canvas.width = Math.ceil( node.canvasBounds.width * scale );
      canvas.height = Math.ceil( node.canvasBounds.height * scale );
      const context = canvas.getContext( '2d' )!;
      const wrapper = new CanvasContextWrapper( canvas, context );
      const matrix = Matrix3.scale( 1 / scale );
      wrapper.context.setTransform( scale, 0, 0, scale, -node.canvasBounds.left, -node.canvasBounds.top );
      node.renderToCanvasSelf( wrapper, matrix );
      setup.url = canvas.toDataURL();
      setup.scale = scale;
      setup.offset = scenerySerialize( node.canvasBounds.leftTop );
    }

    if ( DOM && node instanceof DOM ) {
      serialization.type = 'DOM';
      serialization.element = new window.XMLSerializer().serializeToString( node.element );
      if ( node.element instanceof window.HTMLCanvasElement ) {
        serialization.dataURL = node.element.toDataURL();
      }
      options.preventTransform = node.preventTransform;
    }

    // Paintable
    if ( ( Path && node instanceof Path ) ||
         ( Text && node instanceof Text ) ) {

      [
        'fillPickable',
        'strokePickable',
        'lineWidth',
        'lineCap',
        'lineJoin',
        'lineDashOffset',
        'miterLimit'
      ].forEach( simpleKey => {
        // @ts-expect-error
        if ( node[ simpleKey ] !== PAINTABLE_DEFAULT_OPTIONS[ simpleKey ] ) {
          // @ts-expect-error
          options[ simpleKey ] = node[ simpleKey ];
        }
      } );

      // Ignoring cachedPaints, since we'd 'double' it anyways

      if ( node.fill !== PAINTABLE_DEFAULT_OPTIONS.fill ) {
        setup.fill = scenerySerialize( node.fill );
      }
      if ( node.stroke !== PAINTABLE_DEFAULT_OPTIONS.stroke ) {
        setup.stroke = scenerySerialize( node.stroke );
      }
      if ( node.lineDash.length ) {
        setup.lineDash = scenerySerialize( node.lineDash );
      }
    }

    return serialization;
  }
  else if ( value instanceof Display ) {
    return {
      type: 'Display',
      width: value.width,
      height: value.height,
      backgroundColor: scenerySerialize( value.backgroundColor ),
      tree: {
        type: 'Subtree',
        rootNodeId: value.rootNode.id,
        nodes: serializeConnectedNodes( value.rootNode )
      }
    };
  }
  else {
    return {
      type: 'value',
      value: value
    };
  }
};

const serializeConnectedNodes = ( rootNode: Node ): IntentionalAny => {
  return rootNode.getSubtreeNodes().map( scenerySerialize );
};

scenery.register( 'scenerySerialize', scenerySerialize );
export { scenerySerialize as default, serializeConnectedNodes };