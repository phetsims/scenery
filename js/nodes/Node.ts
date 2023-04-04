// Copyright 2012-2023, University of Colorado Boulder

/**
 * A Node for the Scenery scene graph. Supports general directed acyclic graphics (DAGs).
 * Handles multiple layers with assorted types (Canvas 2D, SVG, DOM, WebGL, etc.).
 *
 * ## General description of Nodes
 *
 * In Scenery, the visual output is determined by a group of connected Nodes (generally known as a scene graph).
 * Each Node has a list of 'child' Nodes. When a Node is visually displayed, its child Nodes (children) will also be
 * displayed, along with their children, etc. There is typically one 'root' Node that is passed to the Scenery Display
 * whose descendants (Nodes that can be traced from the root by child relationships) will be displayed.
 *
 * For instance, say there are Nodes named A, B, C, D and E, who have the relationships:
 * - B is a child of A (thus A is a parent of B)
 * - C is a child of A (thus A is a parent of C)
 * - D is a child of C (thus C is a parent of D)
 * - E is a child of C (thus C is a parent of E)
 * where A would be the root Node. This can be visually represented as a scene graph, where a line connects a parent
 * Node to a child Node (where the parent is usually always at the top of the line, and the child is at the bottom):
 * For example:
 *
 *   A
 *  / \
 * B   C
 *    / \
 *   D   E
 *
 * Additionally, in this case:
 * - D is a 'descendant' of A (due to the C being a child of A, and D being a child of C)
 * - A is an 'ancestor' of D (due to the reverse)
 * - C's 'subtree' is C, D and E, which consists of C itself and all of its descendants.
 *
 * Note that Scenery allows some more complicated forms, where Nodes can have multiple parents, e.g.:
 *
 *   A
 *  / \
 * B   C
 *  \ /
 *   D
 *
 * In this case, D has two parents (B and C). Scenery disallows any Node from being its own ancestor or descendant,
 * so that loops are not possible. When a Node has two or more parents, it means that the Node's subtree will typically
 * be displayed twice on the screen. In the above case, D would appear both at B's position and C's position. Each
 * place a Node would be displayed is known as an 'instance'.
 *
 * Each Node has a 'transform' associated with it, which determines how its subtree (that Node and all of its
 * descendants) will be positioned. Transforms can contain:
 * - Translation, which moves the position the subtree is displayed
 * - Scale, which makes the displayed subtree larger or smaller
 * - Rotation, which displays the subtree at an angle
 * - or any combination of the above that uses an affine matrix (more advanced transforms with shear and combinations
 *   are possible).
 *
 * Say we have the following scene graph:
 *
 *   A
 *   |
 *   B
 *   |
 *   C
 *
 * where there are the following transforms:
 * - A has a 'translation' that moves the content 100 pixels to the right
 * - B has a 'scale' that doubles the size of the content
 * - C has a 'rotation' that rotates 180-degrees around the origin
 *
 * If C displays a square that fills the area with 0 <= x <= 10 and 0 <= y <= 10, we can determine the position on
 * the display by applying transforms starting at C and moving towards the root Node (in this case, A):
 * 1. We apply C's rotation to our square, so the filled area will now be -10 <= x <= 0 and -10 <= y <= 0
 * 2. We apply B's scale to our square, so now we have -20 <= x <= 0 and -20 <= y <= 0
 * 3. We apply A's translation to our square, moving it to 80 <= x <= 100 and -20 <= y <= 0
 *
 * Nodes also have a large number of properties that will affect how their entire subtree is rendered, such as
 * visibility, opacity, etc.
 *
 * ## Creating Nodes
 *
 * Generally, there are two types of Nodes:
 * - Nodes that don't display anything, but serve as a container for other Nodes (e.g. Node itself, HBox, VBox)
 * - Nodes that display content, but ALSO serve as a container (e.g. Circle, Image, Text)
 *
 * When a Node is created with the default Node constructor, e.g.:
 *   var node = new Node();
 * then that Node will not display anything by itself.
 *
 * Generally subtypes of Node are used for displaying things, such as Circle, e.g.:
 *   var circle = new Circle( 20 ); // radius of 20
 *
 * Almost all Nodes (with the exception of leaf-only Nodes like Spacer) can contain children.
 *
 * ## Connecting Nodes, and rendering order
 *
 * To make a 'childNode' become a 'parentNode', the typical way is to call addChild():
 *   parentNode.addChild( childNode );
 *
 * To remove this connection, you can call:
 *   parentNode.removeChild( childNode );
 *
 * Adding a child Node with addChild() puts it at the end of parentNode's list of child Nodes. This is important,
 * because the order of children affects what Nodes are drawn on the 'top' or 'bottom' visually. Nodes that are at the
 * end of the list of children are generally drawn on top.
 *
 * This is generally easiest to represent by notating scene graphs with children in order from left to right, thus:
 *
 *   A
 *  / \
 * B   C
 *    / \
 *   D   E
 *
 * would indicate that A's children are [B,C], so C's subtree is drawn ON TOP of B. The same is true of C's children
 * [D,E], so E is drawn on top of D. If a Node itself has content, it is drawn below that of its children (so C itself
 * would be below D and E).
 *
 * This means that for every scene graph, Nodes instances can be ordered from bottom to top. For the above example, the
 * order is:
 * 1. A (on the very bottom visually, may get covered up by other Nodes)
 * 2. B
 * 3. C
 * 4. D
 * 5. E (on the very top visually, may be covering other Nodes)
 *
 * ## Trails
 *
 * For examples where there are multiple parents for some Nodes (also referred to as DAG in some code, as it represents
 * a Directed Acyclic Graph), we need more information about the rendering order (as otherwise Nodes could appear
 * multiple places in the visual bottom-to-top order.
 *
 * A Trail is basically a list of Nodes, where every Node in the list is a child of its previous element, and a parent
 * of its next element. Thus for the scene graph:
 *
 *   A
 *  / \
 * B   C
 *  \ / \
 *   D   E
 *    \ /
 *     F
 *
 * there are actually three instances of F being displayed, with three trails:
 * - [A,B,D,F]
 * - [A,C,D,F]
 * - [A,C,E,F]
 * Note that the trails are essentially listing Nodes used in walking from the root (A) to the relevant Node (F) using
 * connections between parents and children.
 *
 * The trails above are in order from bottom to top (visually), due to the order of children. Thus since A's children
 * are [B,C] in that order, F with the trail [A,B,D,F] is displayed below [A,C,D,F], because C is after B.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty, { BooleanPropertyOptions } from '../../../axon/js/BooleanProperty.js';
import EnabledProperty, { EnabledPropertyOptions } from '../../../axon/js/EnabledProperty.js';
import Property, { PropertyOptions } from '../../../axon/js/Property.js';
import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import TinyForwardingProperty from '../../../axon/js/TinyForwardingProperty.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import TinyStaticProperty from '../../../axon/js/TinyStaticProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import arrayDifference from '../../../phet-core/js/arrayDifference.js';
import deprecationWarning from '../../../phet-core/js/deprecationWarning.js';
import PhetioObject, { PhetioObjectOptions } from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import BooleanIO from '../../../tandem/js/types/BooleanIO.js';
import IOType from '../../../tandem/js/types/IOType.js';
import TProperty from '../../../axon/js/TProperty.js';
import { ACCESSIBILITY_OPTION_KEYS, CanvasContextWrapper, CanvasSelfDrawable, Display, DOMSelfDrawable, Drawable, Features, Filter, Image, ImageOptions, Instance, isHeightSizable, isWidthSizable, LayoutConstraint, Mouse, ParallelDOM, ParallelDOMOptions, Picker, Pointer, Renderer, RendererSummary, scenery, serializeConnectedNodes, SVGSelfDrawable, TInputListener, TLayoutOptions, Trail, WebGLSelfDrawable } from '../imports.js';
import optionize, { combineOptions, EmptySelfOptions, optionize3 } from '../../../phet-core/js/optionize.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';
import Utils from '../../../dot/js/Utils.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import TEmitter from '../../../axon/js/TEmitter.js';

let globalIdCounter = 1;

const scratchBounds2 = Bounds2.NOTHING.copy(); // mutable {Bounds2} used temporarily in methods
const scratchBounds2Extra = Bounds2.NOTHING.copy(); // mutable {Bounds2} used temporarily in methods
const scratchMatrix3 = new Matrix3();

const ENABLED_PROPERTY_TANDEM_NAME = EnabledProperty.TANDEM_NAME;
const VISIBLE_PROPERTY_TANDEM_NAME = 'visibleProperty';
const INPUT_ENABLED_PROPERTY_TANDEM_NAME = 'inputEnabledProperty';

const PHET_IO_STATE_DEFAULT = false;

// Store the number of parents from the single Node instance that has the most parents in the whole runtime.
let maxParentCount = 0;

export const REQUIRES_BOUNDS_OPTION_KEYS = [
  'leftTop', // {Vector2} - The upper-left corner of this Node's bounds, see setLeftTop() for more documentation
  'centerTop', // {Vector2} - The top-center of this Node's bounds, see setCenterTop() for more documentation
  'rightTop', // {Vector2} - The upper-right corner of this Node's bounds, see setRightTop() for more documentation
  'leftCenter', // {Vector2} - The left-center of this Node's bounds, see setLeftCenter() for more documentation
  'center', // {Vector2} - The center of this Node's bounds, see setCenter() for more documentation
  'rightCenter', // {Vector2} - The center-right of this Node's bounds, see setRightCenter() for more documentation
  'leftBottom', // {Vector2} - The bottom-left of this Node's bounds, see setLeftBottom() for more documentation
  'centerBottom', // {Vector2} - The middle center of this Node's bounds, see setCenterBottom() for more documentation
  'rightBottom', // {Vector2} - The bottom right of this Node's bounds, see setRightBottom() for more documentation
  'left', // {number} - The left side of this Node's bounds, see setLeft() for more documentation
  'right', // {number} - The right side of this Node's bounds, see setRight() for more documentation
  'top', // {number} - The top side of this Node's bounds, see setTop() for more documentation
  'bottom', // {number} - The bottom side of this Node's bounds, see setBottom() for more documentation
  'centerX', // {number} - The x-center of this Node's bounds, see setCenterX() for more documentation
  'centerY' // {number} - The y-center of this Node's bounds, see setCenterY() for more documentation
];

// Node options, in the order they are executed in the constructor/mutate()
const NODE_OPTION_KEYS = [
  'children', // List of children to add (in order), see setChildren for more documentation
  'cursor', // CSS cursor to display when over this Node, see setCursor() for more documentation

  'phetioVisiblePropertyInstrumented', // When true, create an instrumented visibleProperty when this Node is instrumented, see setPhetioVisiblePropertyInstrumented() for more documentation
  'visibleProperty', // Sets forwarding of the visibleProperty, see setVisibleProperty() for more documentation
  'visible', // Whether the Node is visible, see setVisible() for more documentation

  'pickableProperty', // Sets forwarding of the pickableProperty, see setPickableProperty() for more documentation
  'pickable', // Whether the Node is pickable, see setPickable() for more documentation

  'phetioEnabledPropertyInstrumented', // When true, create an instrumented enabledProperty when this Node is instrumented, see setPhetioEnabledPropertyInstrumented() for more documentation
  'enabledProperty', // Sets forwarding of the enabledProperty, see setEnabledProperty() for more documentation
  'enabled', // Whether the Node is enabled, see setEnabled() for more documentation

  'phetioInputEnabledPropertyInstrumented', // When true, create an instrumented inputEnabledProperty when this Node is instrumented, see setPhetioInputEnabledPropertyInstrumented() for more documentation
  'inputEnabledProperty', // Sets forwarding of the inputEnabledProperty, see setInputEnabledProperty() for more documentation
  'inputEnabled', // {boolean} Whether input events can reach into this subtree, see setInputEnabled() for more documentation
  'inputListeners', // The input listeners attached to the Node, see setInputListeners() for more documentation
  'opacity', // Opacity of this Node's subtree, see setOpacity() for more documentation
  'disabledOpacity', // A multiplier to the opacity of this Node's subtree when the node is disabled, see setDisabledOpacity() for more documentation
  'filters', // Non-opacity filters, see setFilters() for more documentation
  'matrix', // Transformation matrix of the Node, see setMatrix() for more documentation
  'translation', // x/y translation of the Node, see setTranslation() for more documentation
  'x', // x translation of the Node, see setX() for more documentation
  'y', // y translation of the Node, see setY() for more documentation
  'rotation', // rotation (in radians) of the Node, see setRotation() for more documentation
  'scale', // scale of the Node, see scale() for more documentation
  'excludeInvisibleChildrenFromBounds', // Controls bounds depending on child visibility, see setExcludeInvisibleChildrenFromBounds() for more documentation
  'layoutOptions', // Provided to layout containers for options, see setLayoutOptions() for more documentation
  'localBounds', // bounds of subtree in local coordinate frame, see setLocalBounds() for more documentation
  'maxWidth', // Constrains width of this Node, see setMaxWidth() for more documentation
  'maxHeight', // Constrains height of this Node, see setMaxHeight() for more documentation
  'renderer', // The preferred renderer for this subtree, see setRenderer() for more documentation
  'layerSplit', // Forces this subtree into a layer of its own, see setLayerSplit() for more documentation
  'usesOpacity', // Hint that opacity will be changed, see setUsesOpacity() for more documentation
  'cssTransform', // Hint that can trigger using CSS transforms, see setCssTransform() for more documentation
  'excludeInvisible', // If this is invisible, exclude from DOM, see setExcludeInvisible() for more documentation
  'webglScale', // Hint to adjust WebGL scaling quality for this subtree, see setWebglScale() for more documentation
  'preventFit', // Prevents layers from fitting this subtree, see setPreventFit() for more documentation
  'mouseArea', // Changes the area the mouse can interact with, see setMouseArea() for more documentation
  'touchArea', // Changes the area touches can interact with, see setTouchArea() for more documentation
  'clipArea', // Makes things outside of a shape invisible, see setClipArea() for more documentation
  'transformBounds', // Flag that makes bounds tighter, see setTransformBounds() for more documentation
  ...REQUIRES_BOUNDS_OPTION_KEYS
];

const DEFAULT_OPTIONS = {
  phetioVisiblePropertyInstrumented: true,
  visible: true,
  opacity: 1,
  disabledOpacity: 1,
  pickable: null,
  enabled: true,
  phetioEnabledPropertyInstrumented: false,
  inputEnabled: true,
  phetioInputEnabledPropertyInstrumented: false,
  clipArea: null,
  mouseArea: null,
  touchArea: null,
  cursor: null,
  transformBounds: false,
  maxWidth: null,
  maxHeight: null,
  renderer: null,
  usesOpacity: false,
  layerSplit: false,
  cssTransform: false,
  excludeInvisible: false,
  webglScale: null,
  preventFit: false
};

export type RendererType = 'svg' | 'canvas' | 'webgl' | 'dom' | null;

// Isolated so that we can delay options that are based on bounds of the Node to after construction.
// See https://github.com/phetsims/scenery/issues/1332
export type NodeBoundsBasedTranslationOptions = {
  leftTop?: Vector2;
  centerTop?: Vector2;
  rightTop?: Vector2;
  leftCenter?: Vector2;
  center?: Vector2;
  rightCenter?: Vector2;
  leftBottom?: Vector2;
  centerBottom?: Vector2;
  rightBottom?: Vector2;
  left?: number;
  right?: number;
  top?: number;
  bottom?: number;
  centerX?: number;
  centerY?: number;
};

// All translation options (includes those based on bounds and those that are not)
export type NodeTranslationOptions = {
  translation?: Vector2;
  x?: number;
  y?: number;
} & NodeBoundsBasedTranslationOptions;

// All transform options (includes translation options)
export type NodeTransformOptions = {
  matrix?: Matrix3;
  rotation?: number;
  scale?: number | Vector2;
} & NodeTranslationOptions;

// All base Node options
export type NodeOptions = {
  children?: Node[];
  cursor?: string | null;
  phetioVisiblePropertyInstrumented?: boolean;
  visibleProperty?: TReadOnlyProperty<boolean> | null;
  visible?: boolean;
  pickableProperty?: TReadOnlyProperty<boolean | null> | null;
  pickable?: boolean | null;
  phetioEnabledPropertyInstrumented?: boolean;
  enabledProperty?: TReadOnlyProperty<boolean> | null;
  enabled?: boolean;
  phetioInputEnabledPropertyInstrumented?: boolean;
  inputEnabledProperty?: TReadOnlyProperty<boolean> | null;
  inputEnabled?: boolean;
  inputListeners?: TInputListener[];
  opacity?: number;
  disabledOpacity?: number;
  filters?: Filter[];
  excludeInvisibleChildrenFromBounds?: boolean;
  layoutOptions?: TLayoutOptions | null;
  localBounds?: Bounds2 | null;
  maxWidth?: number | null;
  maxHeight?: number | null;
  renderer?: RendererType;
  layerSplit?: boolean;
  usesOpacity?: boolean;
  cssTransform?: boolean;
  excludeInvisible?: boolean;
  webglScale?: number | null;
  preventFit?: boolean;
  mouseArea?: Shape | Bounds2 | null;
  touchArea?: Shape | Bounds2 | null;
  clipArea?: Shape | null;
  transformBounds?: boolean;

  // This option is used to create the instrumented, default PhET-iO visibleProperty. These options should not
  // be provided if a `visibleProperty` was provided to this Node, though if they are, they will just be ignored.
  // This grace is to support default options across the component hierarchy melding with usages providing a visibleProperty.
  // This option is a bit buried because it can only be used when the Node is being instrumented, which is when
  // the default, instrumented visibleProperty is conditionally created. We don't want to store these on the Node,
  // and thus they aren't support through `mutate()`.
  visiblePropertyOptions?: PropertyOptions<boolean>;
  enabledPropertyOptions?: PropertyOptions<boolean>;
  inputEnabledPropertyOptions?: PropertyOptions<boolean>;
} & ParallelDOMOptions & NodeTransformOptions;

type RasterizedOptions = {
  resolution?: number;
  sourceBounds?: Bounds2 | null;
  useTargetBounds?: boolean;
  wrap?: boolean;
  useCanvas?: boolean;
  imageOptions?: ImageOptions;
};

class Node extends ParallelDOM {
  // NOTE: All member properties with names starting with '_' are assumed to be private/protected!

  // Assigns a unique ID to this Node (allows trails to get a unique list of IDs)
  public _id: number;

  // All of the Instances tracking this Node
  private readonly _instances: Instance[];

  // All displays where this Node is the root. (scenery-internal)
  public readonly _rootedDisplays: Display[];

  // Drawable states that need to be updated on mutations. Generally added by SVG and
  // DOM elements that need to closely track state (possibly by Canvas to maintain dirty state).
  // (scenery-internal)
  public readonly _drawables: Drawable[];

  // Whether this Node (and its children) will be visible when the scene is updated.
  // Visible Nodes by default will not be pickable either.
  // NOTE: This is fired synchronously when the visibility of the Node is toggled
  private readonly _visibleProperty: TinyForwardingProperty<boolean>;

  // Opacity, in the range from 0 (fully transparent) to 1 (fully opaque).
  // NOTE: This is fired synchronously when the opacity of the Node is toggled
  public readonly opacityProperty: TinyProperty<number>;

  // Disabled opacity, in the range from 0 (fully transparent) to 1 (fully opaque).
  // Combined with the normal opacity ONLY when the node is disabled.
  // NOTE: This is fired synchronously when the opacity of the Node is toggled
  public readonly disabledOpacityProperty: TinyProperty<number>;

  // See setPickable() and setPickableProperty()
  // NOTE: This is fired synchronously when the pickability of the Node is toggled
  private readonly _pickableProperty: TinyForwardingProperty<boolean | null>;

  // See setEnabled() and setEnabledProperty()
  private readonly _enabledProperty: TinyForwardingProperty<boolean>;

  // Whether input event listeners on this Node or descendants on a trail will have
  // input listeners. triggered. Note that this does NOT effect picking, and only prevents some listeners from being
  // fired.
  private readonly _inputEnabledProperty: TinyForwardingProperty<boolean>;

  // This Node and all children will be clipped by this shape (in addition to any
  // other clipping shapes). The shape should be in the local coordinate frame.
  // NOTE: This is fired synchronously when the clipArea of the Node is toggled
  public readonly clipAreaProperty: TinyProperty<Shape | null>;

  // Whether this Node and its subtree can announce content with Voicing and SpeechSynthesis. Though
  // related to Voicing it exists in Node because it is useful to set voicingVisible on a subtree where the
  // root does not compose Voicing. This is not ideal but the entirety of Voicing cannot be composed into every
  // Node because it would produce incorrect behaviors and have a massive memory footprint. See setVoicingVisible()
  // and Voicing.ts for more information about Voicing.
  public readonly voicingVisibleProperty: TinyProperty<boolean>;

  // Areas for hit intersection. If set on a Node, no descendants can handle events.
  // (scenery-internal)
  public _mouseArea: Shape | Bounds2 | null; // for mouse position in the local coordinate frame
  public _touchArea: Shape | Bounds2 | null; // for touch and pen position in the local coordinate frame

  // The CSS cursor to be displayed over this Node. null should be the default (inherit) value.
  private _cursor: string | null;

  // Ordered array of child Nodes.
  // (scenery-internal)
  public _children: Node[];

  // Unordered array of parent Nodes.
  // (scenery-internal)
  public _parents: Node[];

  // Whether we will do more accurate (and tight) bounds computations for rotations and shears.
  private _transformBounds: boolean;

  // Set up the transform reference. we add a listener so that the transform itself can be modified directly
  // by reference, triggering the event notifications for Scenery The reference to the Transform3 will never change.
  // (scenery-internal)
  public _transform: Transform3;
  public _transformListener: () => void;

  // Maximum dimensions for the Node's local bounds before a corrective scaling factor is applied to maintain size.
  // The maximum dimensions are always compared to local bounds, and applied "before" the Node's transform.
  // Whenever the local bounds or maximum dimensions of this Node change and it has at least one maximum dimension
  // (width or height), an ideal scale is computed (either the smallest scale for our local bounds to fit the
  // dimension constraints, OR 1, whichever is lower). Then the Node's transform will be scaled (prepended) with
  // a scale adjustment of ( idealScale / alreadyAppliedScaleFactor ).
  // In the simple case where the Node isn't otherwise transformed, this will apply and update the Node's scale so that
  // the Node matches the maximum dimensions, while never scaling over 1. Note that manually applying transforms to
  // the Node is fine, but may make the Node's width greater than the maximum width.
  // NOTE: If a dimension constraint is null, no resizing will occur due to it. If both maxWidth and maxHeight are null,
  // no scale adjustment will be applied.
  //
  // Also note that setting maxWidth/maxHeight is like adding a local bounds listener (will trigger validation of
  // bounds during the updateDisplay step). NOTE: this means updates to the transform (on a local bounds change) will
  // happen when bounds are validated (validateBounds()), which does not happen synchronously on a child's size
  // change. It does happen at least once in updateDisplay() before rendering, and calling validateBounds() can force
  // a re-check and transform.
  private _maxWidth: number | null;
  private _maxHeight: number | null;

  // Scale applied due to the maximum dimension constraints.
  private _appliedScaleFactor: number;

  // For user input handling (mouse/touch). (scenery-internal)
  public _inputListeners: TInputListener[];

  // [mutable] Bounds for this Node and its children in the "parent" coordinate frame.
  // NOTE: The reference here will not change, we will just notify using the equivalent static notification method.
  // NOTE: This is fired **asynchronously** (usually as part of a Display.updateDisplay()) when the bounds of the Node
  // is changed.
  public readonly boundsProperty: TinyStaticProperty<Bounds2>;

  // [mutable] Bounds for this Node and its children in the "local" coordinate frame.
  // NOTE: The reference here will not change, we will just notify using the equivalent static notification method.
  // NOTE: This is fired **asynchronously** (usually as part of a Display.updateDisplay()) when the localBounds of
  // the Node is changed.
  public readonly localBoundsProperty: TinyStaticProperty<Bounds2>;

  // [mutable] Bounds just for children of this Node (and sub-trees), in the "local" coordinate frame.
  // NOTE: The reference here will not change, we will just notify using the equivalent static notification method.
  // NOTE: This is fired **asynchronously** (usually as part of a Display.updateDisplay()) when the childBounds of the
  // Node is changed.
  public readonly childBoundsProperty: TinyStaticProperty<Bounds2>;

  // [mutable] Bounds just for this Node, in the "local" coordinate frame.
  // NOTE: The reference here will not change, we will just notify using the equivalent static notification method.
  // NOTE: This event can be fired synchronously, and happens with the self-bounds of a Node is changed. This is NOT
  // like the other bounds Properties, which usually fire asynchronously
  public readonly selfBoundsProperty: TinyStaticProperty<Bounds2>;

  // Whether our localBounds have been set (with the ES5 setter/setLocalBounds()) to a custom
  // overridden value. If true, then localBounds itself will not be updated, but will instead always be the
  // overridden value.
  // (scenery-internal)
  public _localBoundsOverridden: boolean;

  // [mutable] Whether invisible children will be excluded from this Node's bounds
  private _excludeInvisibleChildrenFromBounds: boolean;

  // Options that can be provided to layout managers to adjust positioning for this node.
  private _layoutOptions: TLayoutOptions | null;

  // Whether bounds needs to be recomputed to be valid.
  // (scenery-internal)
  public _boundsDirty: boolean;

  // Whether localBounds needs to be recomputed to be valid.
  // (scenery-internal)
  public _localBoundsDirty: boolean;

  // Whether selfBounds needs to be recomputed to be valid.
  // (scenery-internal)
  public _selfBoundsDirty: boolean;

  // Whether childBounds needs to be recomputed to be valid.
  // (scenery-internal)
  public _childBoundsDirty: boolean;

  // (scenery-internal)
  public _filters: Filter[];

  private _originalBounds?: Bounds2; // If assertions are enabled
  private _originalLocalBounds?: Bounds2; // If assertions are enabled
  private _originalSelfBounds?: Bounds2; // If assertions are enabled
  private _originalChildBounds?: Bounds2; // If assertions are enabled

  // Where rendering-specific settings are stored. They are generally modified internally, so there is no ES5 setter
  // for hints.
  // (scenery-internal)
  public _hints: {
    // What type of renderer should be forced for this Node. Uses the internal bitmask structure declared in
    // scenery.js and Renderer.js.
    renderer: number;

    // Whether it is anticipated that opacity will be switched on. If so, having this set to true will make switching
    // back-and-forth between opacity:1 and other opacities much faster.
    usesOpacity: boolean;

    // Whether layers should be split before and after this Node.
    layerSplit: boolean;

    // Whether this Node and its subtree should handle transforms by using a CSS transform of a div.
    cssTransform: boolean;

    // When rendered as Canvas, whether we should use full (device) resolution on retina-like devices.
    // TODO: ensure that this is working? 0.2 may have caused a regression.
    fullResolution: boolean;

    // Whether SVG (or other) content should be excluded from the DOM tree when invisible (instead of just being hidden)
    excludeInvisible: boolean;

    // If non-null, a multiplier to the detected pixel-to-pixel scaling of the WebGL Canvas
    webglScale: number | null;

    // If true, Scenery will not fit any blocks that contain drawables attached to Nodes underneath this Node's subtree.
    // This will typically prevent Scenery from triggering bounds computation for this sub-tree, and movement of this
    // Node or its descendants will never trigger the refitting of a block.
    preventFit: boolean;
  };

  // This is fired only once for any single operation that may change the children of a Node.
  // For example, if a Node's children are [ a, b ] and setChildren( [ a, x, y, z ] ) is called on it, the
  // childrenChanged event will only be fired once after the entire operation of changing the children is completed.
  public readonly childrenChangedEmitter: TEmitter;

  // For every single added child Node, emits with {Node} Node, {number} indexOfChild
  public readonly childInsertedEmitter: TEmitter<[ node: Node, indexOfChild: number ]>;

  // For every single removed child Node, emits with {Node} Node, {number} indexOfChild
  public readonly childRemovedEmitter: TEmitter<[ node: Node, indexOfChild: number ]>;

  // Provides a given range that may be affected by the reordering
  public readonly childrenReorderedEmitter: TEmitter<[ minChangedIndex: number, maxChangedIndex: number ]>;

  // Fired whenever a parent is added
  public readonly parentAddedEmitter: TEmitter<[ node: Node ]> = new TinyEmitter();

  // Fired whenever a parent is removed
  public readonly parentRemovedEmitter: TEmitter<[ node: Node ]> = new TinyEmitter();

  // Fired synchronously when the transform (transformation matrix) of a Node is changed. Any
  // change to a Node's translation/rotation/scale/etc. will trigger this event.
  public readonly transformEmitter: TEmitter;

  // Should be emitted when we need to check full metadata updates directly on Instances,
  // to see if we need to change drawable types, etc.
  public readonly instanceRefreshEmitter: TEmitter;

  // Emitted to when we need to potentially recompute our renderer summary (bitmask flags, or
  // things that could affect descendants)
  public readonly rendererSummaryRefreshEmitter: TEmitter;

  // Emitted to when we change filters (either opacity or generalized filters)
  public readonly filterChangeEmitter: TEmitter;

  // Fired when an instance is changed (added/removed)
  public readonly changedInstanceEmitter: TEmitter<[ instance: Instance, added: boolean ]>;

  // Fired when layoutOptions changes
  public readonly layoutOptionsChangedEmitter: TEmitter;

  // A bitmask which specifies which renderers this Node (and only this Node, not its subtree) supports.
  // (scenery-internal)
  public _rendererBitmask: number;

  // A bitmask-like summary of what renderers and options are supported by this Node and all of its descendants
  // (scenery-internal)
  public _rendererSummary: RendererSummary;

  // So we can traverse only the subtrees that require bounds validation for events firing.
  // This is a sum of the number of events requiring bounds validation on this Node, plus the number of children whose
  // count is non-zero.
  // NOTE: this means that if A has a child B, and B has a boundsEventCount of 5, it only contributes 1 to A's count.
  // This allows us to have changes localized (increasing B's count won't change A or any of A's ancestors), and
  // guarantees that we will know whether a subtree has bounds listeners. Also important: decreasing B's
  // boundsEventCount down to 0 will allow A to decrease its count by 1, without having to check its other children
  // (if we were just using a boolean value, this operation would require A to check if any OTHER children besides
  // B had bounds listeners)
  // (scenery-internal)
  public _boundsEventCount: number;

  // This signals that we can validateBounds() on this subtree and we don't have to traverse further
  // (scenery-internal)
  public _boundsEventSelfCount: number;

  // Subcomponent dedicated to hit testing
  // (scenery-internal)
  public _picker: Picker;

  // There are certain specific cases (in this case due to a11y) where we need
  // to know that a Node is getting removed from its parent BUT that process has not completed yet. It would be ideal
  // to not need this.
  // (scenery-internal)
  public _isGettingRemovedFromParent: boolean;

  // {Object} - A mapping of all of options that require Bounds to be applied properly. Most often these should be set through `mutate` in the end of the construcor instead of being passed through `super()`
  public static readonly REQUIRES_BOUNDS_OPTION_KEYS = REQUIRES_BOUNDS_OPTION_KEYS;

  // Used by sceneryDeserialize
  // (scenery-internal)
  public _serialization?: IntentionalAny;

  // Tracks any layout constraint, so that we can avoid having multiple layout constraints on the same node
  // (and avoid the infinite loops that can happen if that is triggered).
  // (scenery-internal)
  public _activeParentLayoutConstraint: LayoutConstraint | null = null;

  // This is an array of property (setter) names for Node.mutate(), which are also used when creating
  // Nodes with parameter objects.
  //
  // E.g. new phet.scenery.Node( { x: 5, rotation: 20 } ) will create a Path, and apply setters in the order below
  // (node.x = 5; node.rotation = 20)
  //
  // Some special cases exist (for function names). new phet.scenery.Node( { scale: 2 } ) will actually call
  // node.scale( 2 ).
  //
  // The order below is important! Don't change this without knowing the implications.
  //
  // NOTE: Translation-based mutators come before rotation/scale, since typically we think of their operations
  //       occurring "after" the rotation / scaling
  // NOTE: left/right/top/bottom/centerX/centerY are at the end, since they rely potentially on rotation / scaling
  //       changes of bounds that may happen beforehand
  // (scenery-internal)
  public _mutatorKeys!: string[];

  // List of all dirty flags that should be available on drawables created from this Node (or
  // subtype). Given a flag (e.g. radius), it indicates the existence of a function
  // drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
  // (scenery-internal)
  //
  // Should be overridden by subtypes.
  public drawableMarkFlags!: string[];

  /**
   * Creates a Node with options.
   *
   * NOTE: Directly created Nodes (not of any subtype, but created with "new Node( ... )") are generally used as
   *       containers, which can hold other Nodes, subtypes of Node that can display things.
   *
   * Node and its subtypes generally have the last constructor parameter reserved for the 'options' object. This is a
   * key-value map that specifies relevant options that are used by Node and subtypes.
   *
   * For example, one of Node's options is bottom, and one of Circle's options is radius. When a circle is created:
   *   var circle = new Circle( {
   *     radius: 10,
   *     bottom: 200
   *   } );
   * This will create a Circle, set its radius (by executing circle.radius = 10, which uses circle.setRadius()), and
   * then will align the bottom of the circle along y=200 (by executing circle.bottom = 200, which uses
   * node.setBottom()).
   *
   * The options are executed in the order specified by each types _mutatorKeys property.
   *
   * The options object is currently not checked to see whether there are property (key) names that are not used, so it
   * is currently legal to do "new Node( { fork_kitchen_spoon: 5 } )".
   *
   * Usually, an option (e.g. 'visible'), when used in a constructor or mutate() call, will directly use the ES5 setter
   * for that property (e.g. node.visible = ...), which generally forwards to a non-ES5 setter function
   * (e.g. node.setVisible( ... )) that is responsible for the behavior. Documentation is generally on these methods
   * (e.g. setVisible), although some methods may be dynamically created to avoid verbosity (like node.leftTop).
   *
   * Sometimes, options invoke a function instead (e.g. 'scale') because the verb and noun are identical. In this case,
   * instead of setting the setter (node.scale = ..., which would override the function), it will instead call
   * the method directly (e.g. node.scale( ... )).
   */
  public constructor( options?: NodeOptions ) {

    super();

    this._id = globalIdCounter++;
    this._instances = [];
    this._rootedDisplays = [];
    this._drawables = [];
    this._visibleProperty = new TinyForwardingProperty( DEFAULT_OPTIONS.visible, DEFAULT_OPTIONS.phetioVisiblePropertyInstrumented,
      this.onVisiblePropertyChange.bind( this ) );
    this.opacityProperty = new TinyProperty( DEFAULT_OPTIONS.opacity, this.onOpacityPropertyChange.bind( this ) );
    this.disabledOpacityProperty = new TinyProperty( DEFAULT_OPTIONS.disabledOpacity, this.onDisabledOpacityPropertyChange.bind( this ) );
    this._pickableProperty = new TinyForwardingProperty<boolean | null>( DEFAULT_OPTIONS.pickable,
      false, this.onPickablePropertyChange.bind( this ) );
    this._enabledProperty = new TinyForwardingProperty<boolean>( DEFAULT_OPTIONS.enabled,
      DEFAULT_OPTIONS.phetioEnabledPropertyInstrumented, this.onEnabledPropertyChange.bind( this ) );

    this._inputEnabledProperty = new TinyForwardingProperty( DEFAULT_OPTIONS.inputEnabled,
      DEFAULT_OPTIONS.phetioInputEnabledPropertyInstrumented );
    this.clipAreaProperty = new TinyProperty<Shape | null>( DEFAULT_OPTIONS.clipArea );
    this.voicingVisibleProperty = new TinyProperty<boolean>( true );
    this._mouseArea = DEFAULT_OPTIONS.mouseArea;
    this._touchArea = DEFAULT_OPTIONS.touchArea;
    this._cursor = DEFAULT_OPTIONS.cursor;
    this._children = [];
    this._parents = [];
    this._transformBounds = DEFAULT_OPTIONS.transformBounds;
    this._transform = new Transform3();
    this._transformListener = this.onTransformChange.bind( this );
    this._transform.changeEmitter.addListener( this._transformListener );
    this._maxWidth = DEFAULT_OPTIONS.maxWidth;
    this._maxHeight = DEFAULT_OPTIONS.maxHeight;
    this._appliedScaleFactor = 1;
    this._inputListeners = [];

    this.inputEnabledProperty.lazyLink( this.pdomBoundInputEnabledListener );

    // Add listener count change notifications into these Properties, since we need to know when their number of listeners
    // changes dynamically.
    const boundsListenersAddedOrRemovedListener = this.onBoundsListenersAddedOrRemoved.bind( this );

    const boundsInvalidationListener = this.validateBounds.bind( this );
    const selfBoundsInvalidationListener = this.validateSelfBounds.bind( this );

    this.boundsProperty = new TinyStaticProperty( Bounds2.NOTHING.copy(), boundsInvalidationListener );
    this.boundsProperty.changeCount = boundsListenersAddedOrRemovedListener;

    this.localBoundsProperty = new TinyStaticProperty( Bounds2.NOTHING.copy(), boundsInvalidationListener );
    this.localBoundsProperty.changeCount = boundsListenersAddedOrRemovedListener;

    this.childBoundsProperty = new TinyStaticProperty( Bounds2.NOTHING.copy(), boundsInvalidationListener );
    this.childBoundsProperty.changeCount = boundsListenersAddedOrRemovedListener;

    this.selfBoundsProperty = new TinyStaticProperty( Bounds2.NOTHING.copy(), selfBoundsInvalidationListener );

    this._localBoundsOverridden = false;
    this._excludeInvisibleChildrenFromBounds = false;
    this._layoutOptions = null;
    this._boundsDirty = true;
    this._localBoundsDirty = true;
    this._selfBoundsDirty = true;
    this._childBoundsDirty = true;

    if ( assert ) {
      // for assertions later to ensure that we are using the same Bounds2 copies as before
      this._originalBounds = this.boundsProperty._value;
      this._originalLocalBounds = this.localBoundsProperty._value;
      this._originalSelfBounds = this.selfBoundsProperty._value;
      this._originalChildBounds = this.childBoundsProperty._value;
    }

    this._filters = [];

    this._hints = {
      renderer: DEFAULT_OPTIONS.renderer === null ? 0 : Renderer.fromName( DEFAULT_OPTIONS.renderer ),
      usesOpacity: DEFAULT_OPTIONS.usesOpacity,
      layerSplit: DEFAULT_OPTIONS.layerSplit,
      cssTransform: DEFAULT_OPTIONS.cssTransform,
      fullResolution: false,
      excludeInvisible: DEFAULT_OPTIONS.excludeInvisible,
      webglScale: DEFAULT_OPTIONS.webglScale,
      preventFit: DEFAULT_OPTIONS.preventFit
    };

    this.childrenChangedEmitter = new TinyEmitter();
    this.childInsertedEmitter = new TinyEmitter();
    this.childRemovedEmitter = new TinyEmitter();
    this.childrenReorderedEmitter = new TinyEmitter();
    this.transformEmitter = new TinyEmitter();
    this.instanceRefreshEmitter = new TinyEmitter();
    this.rendererSummaryRefreshEmitter = new TinyEmitter();
    this.filterChangeEmitter = new TinyEmitter();
    this.changedInstanceEmitter = new TinyEmitter();
    this.layoutOptionsChangedEmitter = new TinyEmitter();

    this._rendererBitmask = Renderer.bitmaskNodeDefault;
    this._rendererSummary = new RendererSummary( this );

    this._boundsEventCount = 0;
    this._boundsEventSelfCount = 0;
    this._picker = new Picker( this );
    this._isGettingRemovedFromParent = false;

    if ( options ) {
      this.mutate( options );
    }
  }


  /**
   * Inserts a child Node at a specific index.
   *
   * node.insertChild( 0, childNode ) will insert the child into the beginning of the children array (on the bottom
   * visually).
   *
   * node.insertChild( node.children.length, childNode ) is equivalent to node.addChild( childNode ), and appends it
   * to the end (top visually) of the children array. It is recommended to use node.addChild when possible.
   *
   * NOTE: overridden by Leaf for some subtypes
   *
   * @param index - Index where the inserted child Node will be after this operation.
   * @param node - The new child to insert.
   * @param [isComposite] - (scenery-internal) If true, the childrenChanged event will not be sent out.
   */
  public insertChild( index: number, node: Node, isComposite?: boolean ): this {
    assert && assert( node !== null && node !== undefined, 'insertChild cannot insert a null/undefined child' );
    assert && assert( !_.includes( this._children, node ), 'Parent already contains child' );
    assert && assert( node !== this, 'Cannot add self as a child' );
    assert && assert( node._parents !== null, 'Tried to insert a disposed child node?' );
    assert && assert( !node.isDisposed, 'Tried to insert a disposed Node' );

    // needs to be early to prevent re-entrant children modifications
    this._picker.onInsertChild( node );
    this.changeBoundsEventCount( node._boundsEventCount > 0 ? 1 : 0 );
    this._rendererSummary.summaryChange( RendererSummary.bitmaskAll, node._rendererSummary.bitmask );

    node._parents.push( this );
    if ( assert && window.phet?.chipper?.queryParameters && isFinite( phet.chipper.queryParameters.parentLimit ) ) {
      const parentCount = node._parents.length;
      if ( maxParentCount < parentCount ) {
        maxParentCount = parentCount;
        console.log( `Max Node parents: ${maxParentCount}` );
        assert( maxParentCount <= phet.chipper.queryParameters.parentLimit,
          `parent count of ${maxParentCount} above ?parentLimit=${phet.chipper.queryParameters.parentLimit}` );
      }
    }
    this._children.splice( index, 0, node );

    // If this added subtree contains PDOM content, we need to notify any relevant displays
    if ( !node._rendererSummary.hasNoPDOM() ) {
      this.onPDOMAddChild( node );
    }

    node.invalidateBounds();

    // like calling this.invalidateBounds(), but we already marked all ancestors with dirty child bounds
    this._boundsDirty = true;

    this.childInsertedEmitter.emit( node, index );
    node.parentAddedEmitter.emit( this );

    !isComposite && this.childrenChangedEmitter.emit();

    if ( assertSlow ) { this._picker.audit(); }

    return this; // allow chaining
  }

  /**
   * Appends a child Node to our list of children.
   *
   * The new child Node will be displayed in front (on top) of all of this node's other children.
   *
   * @param node
   * @param [isComposite] - (scenery-internal) If true, the childrenChanged event will not be sent out.
   */
  public addChild( node: Node, isComposite?: boolean ): this {
    this.insertChild( this._children.length, node, isComposite );

    return this; // allow chaining
  }

  /**
   * Removes a child Node from our list of children, see http://phetsims.github.io/scenery/doc/#node-removeChild
   * Will fail an assertion if the Node is not currently one of our children
   *
   * @param node
   * @param [isComposite] - (scenery-internal) If true, the childrenChanged event will not be sent out.
   */
  public removeChild( node: Node, isComposite?: boolean ): this {
    assert && assert( node && node instanceof Node, 'Need to call node.removeChild() with a Node.' );
    assert && assert( this.hasChild( node ), 'Attempted to removeChild with a node that was not a child.' );

    const indexOfChild = _.indexOf( this._children, node );

    this.removeChildWithIndex( node, indexOfChild, isComposite );

    return this; // allow chaining
  }

  /**
   * Removes a child Node at a specific index (node.children[ index ]) from our list of children.
   * Will fail if the index is out of bounds.
   *
   * @param index
   * @param [isComposite] - (scenery-internal) If true, the childrenChanged event will not be sent out.
   */
  public removeChildAt( index: number, isComposite?: boolean ): this {
    assert && assert( index >= 0 );
    assert && assert( index < this._children.length );

    const node = this._children[ index ];

    this.removeChildWithIndex( node, index, isComposite );

    return this; // allow chaining
  }

  /**
   * Internal method for removing a Node (always has the Node and index).
   *
   * NOTE: overridden by Leaf for some subtypes
   *
   * @param node - The child node to remove from this Node (it's parent)
   * @param indexOfChild - Should satisfy this.children[ indexOfChild ] === node
   * @param [isComposite] - (scenery-internal) If true, the childrenChanged event will not be sent out.
   */
  public removeChildWithIndex( node: Node, indexOfChild: number, isComposite?: boolean ): void {
    assert && assert( node && node instanceof Node, 'Need to call node.removeChildWithIndex() with a Node.' );
    assert && assert( this.hasChild( node ), 'Attempted to removeChild with a node that was not a child.' );
    assert && assert( this._children[ indexOfChild ] === node, 'Incorrect index for removeChildWithIndex' );
    assert && assert( node._parents !== null, 'Tried to remove a disposed child node?' );

    const indexOfParent = _.indexOf( node._parents, this );

    node._isGettingRemovedFromParent = true;

    // If this added subtree contains PDOM content, we need to notify any relevant displays
    // NOTE: Potentially removes bounds listeners here!
    if ( !node._rendererSummary.hasNoPDOM() ) {
      this.onPDOMRemoveChild( node );
    }

    // needs to be early to prevent re-entrant children modifications
    this._picker.onRemoveChild( node );
    this.changeBoundsEventCount( node._boundsEventCount > 0 ? -1 : 0 );
    this._rendererSummary.summaryChange( node._rendererSummary.bitmask, RendererSummary.bitmaskAll );

    node._parents.splice( indexOfParent, 1 );
    this._children.splice( indexOfChild, 1 );
    node._isGettingRemovedFromParent = false; // It is "complete"

    this.invalidateBounds();
    this._childBoundsDirty = true; // force recomputation of child bounds after removing a child

    this.childRemovedEmitter.emit( node, indexOfChild );
    node.parentRemovedEmitter.emit( this );

    !isComposite && this.childrenChangedEmitter.emit();

    if ( assertSlow ) { this._picker.audit(); }
  }

  /**
   * If a child is not at the given index, it is moved to the given index. This reorders the children of this Node so
   * that `this.children[ index ] === node`.
   *
   * @param node - The child Node to move in the order
   * @param index - The desired index (into the children array) of the child.
   */
  public moveChildToIndex( node: Node, index: number ): this {
    assert && assert( this.hasChild( node ), 'Attempted to moveChildToIndex with a node that was not a child.' );
    assert && assert( index % 1 === 0 && index >= 0 && index < this._children.length,
      `Invalid index: ${index}` );

    const currentIndex = this.indexOfChild( node );
    if ( this._children[ index ] !== node ) {

      // Apply the actual children change
      this._children.splice( currentIndex, 1 );
      this._children.splice( index, 0, node );

      if ( !this._rendererSummary.hasNoPDOM() ) {
        this.onPDOMReorderedChildren();
      }

      this.childrenReorderedEmitter.emit( Math.min( currentIndex, index ), Math.max( currentIndex, index ) );
      this.childrenChangedEmitter.emit();
    }

    return this;
  }

  /**
   * Removes all children from this Node.
   */
  public removeAllChildren(): this {
    this.setChildren( [] );

    return this; // allow chaining
  }

  /**
   * Sets the children of the Node to be equivalent to the passed-in array of Nodes.
   *
   * NOTE: Meant to be overridden in some cases
   */
  public setChildren( children: Node[] ): this {
    // The implementation is split into basically three stages:
    // 1. Remove current children that are not in the new children array.
    // 2. Reorder children that exist both before/after the change.
    // 3. Insert in new children

    const beforeOnly: Node[] = []; // Will hold all nodes that will be removed.
    const afterOnly: Node[] = []; // Will hold all nodes that will be "new" children (added)
    const inBoth: Node[] = []; // Child nodes that "stay". Will be ordered for the "after" case.
    let i;

    // Compute what things were added, removed, or stay.
    arrayDifference( children, this._children, afterOnly, beforeOnly, inBoth );

    // Remove any nodes that are not in the new children.
    for ( i = beforeOnly.length - 1; i >= 0; i-- ) {
      this.removeChild( beforeOnly[ i ], true );
    }

    assert && assert( this._children.length === inBoth.length,
      'Removing children should not have triggered other children changes' );

    // Handle the main reordering (of nodes that "stay")
    let minChangeIndex = -1; // What is the smallest index where this._children[ index ] !== inBoth[ index ]
    let maxChangeIndex = -1; // What is the largest index where this._children[ index ] !== inBoth[ index ]
    for ( i = 0; i < inBoth.length; i++ ) {
      const desired = inBoth[ i ];
      if ( this._children[ i ] !== desired ) {
        this._children[ i ] = desired;
        if ( minChangeIndex === -1 ) {
          minChangeIndex = i;
        }
        maxChangeIndex = i;
      }
    }
    // If our minChangeIndex is still -1, then none of those nodes that "stay" were reordered. It's important to check
    // for this case, so that `node.children = node.children` is effectively a no-op performance-wise.
    const hasReorderingChange = minChangeIndex !== -1;

    // Immediate consequences/updates from reordering
    if ( hasReorderingChange ) {
      if ( !this._rendererSummary.hasNoPDOM() ) {
        this.onPDOMReorderedChildren();
      }

      this.childrenReorderedEmitter.emit( minChangeIndex, maxChangeIndex );
    }

    // Add in "new" children.
    // Scan through the "ending" children indices, adding in things that were in the "afterOnly" part. This scan is
    // done through the children array instead of the afterOnly array (as determining the index in children would
    // then be quadratic in time, which would be unacceptable here). At this point, a forward scan should be
    // sufficient to insert in-place, and should move the least amount of nodes in the array.
    if ( afterOnly.length ) {
      let afterIndex = 0;
      let after = afterOnly[ afterIndex ];
      for ( i = 0; i < children.length; i++ ) {
        if ( children[ i ] === after ) {
          this.insertChild( i, after, true );
          after = afterOnly[ ++afterIndex ];
        }
      }
    }

    // If we had any changes, send the generic "changed" event.
    if ( beforeOnly.length !== 0 || afterOnly.length !== 0 || hasReorderingChange ) {
      this.childrenChangedEmitter.emit();
    }

    // Sanity checks to make sure our resulting children array is correct.
    if ( assert ) {
      for ( let j = 0; j < this._children.length; j++ ) {
        assert( children[ j ] === this._children[ j ],
          'Incorrect child after setChildren, possibly a reentrancy issue' );
      }
    }

    // allow chaining
    return this;
  }

  /**
   * See setChildren() for more information
   */
  public set children( value: Node[] ) {
    this.setChildren( value );
  }

  /**
   * See getChildren() for more information
   */
  public get children(): Node[] {
    return this.getChildren();
  }

  /**
   * Returns a defensive copy of the array of direct children of this node, ordered by what is in front (nodes at
   * the end of the array are in front of nodes at the start).
   *
   * Making changes to the returned result will not affect this node's children.
   */
  public getChildren(): Node[] {
    // TODO: ensure we are not triggering this in Scenery code when not necessary!
    return this._children.slice( 0 ); // create a defensive copy
  }

  /**
   * Returns a count of children, without needing to make a defensive copy.
   */
  public getChildrenCount(): number {
    return this._children.length;
  }

  /**
   * Returns a defensive copy of our parents. This is an array of parent nodes that is returned in no particular
   * order (as order is not important here).
   *
   * NOTE: Modifying the returned array will not in any way modify this node's parents.
   */
  public getParents(): Node[] {
    return this._parents.slice( 0 ); // create a defensive copy
  }

  /**
   * See getParents() for more information
   */
  public get parents(): Node[] {
    return this.getParents();
  }

  /**
   * Returns a single parent if it exists, otherwise null (no parents), or an assertion failure (multiple parents).
   */
  public getParent(): Node | null {
    assert && assert( this._parents.length <= 1, 'Cannot call getParent on a node with multiple parents' );
    return this._parents.length ? this._parents[ 0 ] : null;
  }

  /**
   * See getParent() for more information
   */
  public get parent(): Node | null {
    return this.getParent();
  }

  /**
   * Gets the child at a specific index into the children array.
   */
  public getChildAt( index: number ): Node {
    return this._children[ index ];
  }

  /**
   * Finds the index of a parent Node in the parents array.
   *
   * @param parent - Should be a parent of this node.
   * @returns - An index such that this.parents[ index ] === parent
   */
  public indexOfParent( parent: Node ): number {
    return _.indexOf( this._parents, parent );
  }

  /**
   * Finds the index of a child Node in the children array.
   *
   * @param child - Should be a child of this node.
   * @returns - An index such that this.children[ index ] === child
   */
  public indexOfChild( child: Node ): number {
    return _.indexOf( this._children, child );
  }

  /**
   * Moves this Node to the front (end) of all of its parents children array.
   */
  public moveToFront(): this {
    _.each( this._parents.slice(), parent => parent.moveChildToFront( this ) );

    return this; // allow chaining
  }

  /**
   * Moves one of our children to the front (end) of our children array.
   *
   * @param child - Our child to move to the front.
   */
  public moveChildToFront( child: Node ): this {
    return this.moveChildToIndex( child, this._children.length - 1 );
  }

  /**
   * Move this node one index forward in each of its parents.  If the Node is already at the front, this is a no-op.
   */
  public moveForward(): this {
    this._parents.forEach( parent => parent.moveChildForward( this ) ); // TODO: Do we need slice like moveToFront has?
    return this; // chaining
  }

  /**
   * Moves the specified child forward by one index.  If the child is already at the front, this is a no-op.
   */
  public moveChildForward( child: Node ): this {
    const index = this.indexOfChild( child );
    if ( index < this.getChildrenCount() - 1 ) {
      this.moveChildToIndex( child, index + 1 );
    }
    return this; // chaining
  }

  /**
   * Move this node one index backward in each of its parents.  If the Node is already at the back, this is a no-op.
   */
  public moveBackward(): this {
    this._parents.forEach( parent => parent.moveChildBackward( this ) ); // TODO: Do we need slice like moveToFront has?
    return this; // chaining
  }

  /**
   * Moves the specified child forward by one index.  If the child is already at the back, this is a no-op.
   */
  public moveChildBackward( child: Node ): this {
    const index = this.indexOfChild( child );
    if ( index > 0 ) {
      this.moveChildToIndex( child, index - 1 );
    }
    return this; // chaining
  }

  /**
   * Moves this Node to the back (front) of all of its parents children array.
   */
  public moveToBack(): this {
    _.each( this._parents.slice(), parent => parent.moveChildToBack( this ) );

    return this; // allow chaining
  }

  /**
   * Moves one of our children to the back (front) of our children array.
   *
   * @param child - Our child to move to the back.
   */
  public moveChildToBack( child: Node ): this {
    return this.moveChildToIndex( child, 0 );
  }

  /**
   * Replace a child in this node's children array with another node. If the old child had DOM focus and
   * the new child is focusable, the new child will receive focus after it is added.
   */
  public replaceChild( oldChild: Node, newChild: Node ): this {
    assert && assert( this.hasChild( oldChild ), 'Attempted to replace a node that was not a child.' );

    // information that needs to be restored
    const index = this.indexOfChild( oldChild );
    const oldChildFocused = oldChild.focused;

    this.removeChild( oldChild, true );
    this.insertChild( index, newChild, true );

    this.childrenChangedEmitter.emit();

    if ( oldChildFocused && newChild.focusable ) {
      newChild.focus();
    }

    return this; // allow chaining
  }

  /**
   * Removes this Node from all of its parents.
   */
  public detach(): this {
    _.each( this._parents.slice( 0 ), parent => parent.removeChild( this ) );

    return this; // allow chaining
  }

  /**
   * Update our event count, usually by 1 or -1. See documentation on _boundsEventCount in constructor.
   *
   * @param n - How to increment/decrement the bounds event listener count
   */
  private changeBoundsEventCount( n: number ): void {
    if ( n !== 0 ) {
      const zeroBefore = this._boundsEventCount === 0;

      this._boundsEventCount += n;
      assert && assert( this._boundsEventCount >= 0, 'subtree bounds event count should be guaranteed to be >= 0' );

      const zeroAfter = this._boundsEventCount === 0;

      if ( zeroBefore !== zeroAfter ) {
        // parents will only have their count
        const parentDelta = zeroBefore ? 1 : -1;

        const len = this._parents.length;
        for ( let i = 0; i < len; i++ ) {
          this._parents[ i ].changeBoundsEventCount( parentDelta );
        }
      }
    }
  }

  /**
   * Ensures that the cached selfBounds of this Node is accurate. Returns true if any sort of dirty flag was set
   * before this was called.
   *
   * @returns - Was the self-bounds potentially updated?
   */
  public validateSelfBounds(): boolean {
    // validate bounds of ourself if necessary
    if ( this._selfBoundsDirty ) {
      const oldSelfBounds = scratchBounds2.set( this.selfBoundsProperty._value );

      // Rely on an overloadable method to accomplish computing our self bounds. This should update
      // this.selfBounds itself, returning whether it was actually changed. If it didn't change, we don't want to
      // send a 'selfBounds' event.
      const didSelfBoundsChange = this.updateSelfBounds();
      this._selfBoundsDirty = false;

      if ( didSelfBoundsChange ) {
        this.selfBoundsProperty.notifyListeners( oldSelfBounds );
      }

      return true;
    }

    return false;
  }

  /**
   * Ensures that cached bounds stored on this Node (and all children) are accurate. Returns true if any sort of dirty
   * flag was set before this was called.
   *
   * @returns - Was something potentially updated?
   */
  public validateBounds(): boolean {

    sceneryLog && sceneryLog.bounds && sceneryLog.bounds( `validateBounds #${this._id}` );
    sceneryLog && sceneryLog.bounds && sceneryLog.push();

    let i;
    const notificationThreshold = 1e-13;

    let wasDirtyBefore = this.validateSelfBounds();

    // We're going to directly mutate these instances
    const ourChildBounds = this.childBoundsProperty._value;
    const ourLocalBounds = this.localBoundsProperty._value;
    const ourSelfBounds = this.selfBoundsProperty._value;
    const ourBounds = this.boundsProperty._value;

    // validate bounds of children if necessary
    if ( this._childBoundsDirty ) {
      wasDirtyBefore = true;

      sceneryLog && sceneryLog.bounds && sceneryLog.bounds( 'childBounds dirty' );

      // have each child validate their own bounds
      i = this._children.length;
      while ( i-- ) {
        const child = this._children[ i ];

        // Reentrancy might cause the child to be removed
        if ( child ) {
          child.validateBounds();
        }
      }

      // and recompute our childBounds
      const oldChildBounds = scratchBounds2.set( ourChildBounds ); // store old value in a temporary Bounds2
      ourChildBounds.set( Bounds2.NOTHING ); // initialize to a value that can be unioned with includeBounds()

      i = this._children.length;
      while ( i-- ) {
        const child = this._children[ i ];

        // Reentrancy might cause the child to be removed
        if ( child && !this._excludeInvisibleChildrenFromBounds || child.isVisible() ) {
          ourChildBounds.includeBounds( child.bounds );
        }
      }

      // run this before firing the event
      this._childBoundsDirty = false;
      sceneryLog && sceneryLog.bounds && sceneryLog.bounds( `childBounds: ${ourChildBounds}` );

      if ( !ourChildBounds.equals( oldChildBounds ) ) {
        // notifies only on an actual change
        if ( !ourChildBounds.equalsEpsilon( oldChildBounds, notificationThreshold ) ) {
          this.childBoundsProperty.notifyListeners( oldChildBounds ); // RE-ENTRANT CALL HERE, it will validateBounds()
        }
      }

      // WARNING: Think twice before adding code here below the listener notification. The notifyListeners() call can
      // trigger re-entrancy, so this function needs to work when that happens. DO NOT set things based on local
      // variables here.
    }

    if ( this._localBoundsDirty ) {
      wasDirtyBefore = true;

      sceneryLog && sceneryLog.bounds && sceneryLog.bounds( 'localBounds dirty' );

      this._localBoundsDirty = false; // we only need this to set local bounds as dirty

      const oldLocalBounds = scratchBounds2.set( ourLocalBounds ); // store old value in a temporary Bounds2

      // Only adjust the local bounds if it is not overridden
      if ( !this._localBoundsOverridden ) {
        // local bounds are a union between our self bounds and child bounds
        ourLocalBounds.set( ourSelfBounds ).includeBounds( ourChildBounds );

        // apply clipping to the bounds if we have a clip area (all done in the local coordinate frame)
        const clipArea = this.clipArea;
        if ( clipArea ) {
          ourLocalBounds.constrainBounds( clipArea.bounds );
        }
      }

      sceneryLog && sceneryLog.bounds && sceneryLog.bounds( `localBounds: ${ourLocalBounds}` );

      // NOTE: we need to update max dimensions still even if we are setting overridden localBounds
      // adjust our transform to match maximum bounds if necessary on a local bounds change
      if ( this._maxWidth !== null || this._maxHeight !== null ) {
        // needs to run before notifications below, otherwise reentrancy that hits this codepath will have its
        // updateMaxDimension overridden by the eventual original function call, with the now-incorrect local bounds.
        // See https://github.com/phetsims/joist/issues/725
        this.updateMaxDimension( ourLocalBounds );
      }

      if ( !ourLocalBounds.equals( oldLocalBounds ) ) {
        // sanity check, see https://github.com/phetsims/scenery/issues/1071, we're running this before the localBounds
        // listeners are notified, to support limited re-entrance.
        this._boundsDirty = true;

        if ( !ourLocalBounds.equalsEpsilon( oldLocalBounds, notificationThreshold ) ) {
          this.localBoundsProperty.notifyListeners( oldLocalBounds ); // RE-ENTRANT CALL HERE, it will validateBounds()
        }
      }

      // WARNING: Think twice before adding code here below the listener notification. The notifyListeners() call can
      // trigger re-entrancy, so this function needs to work when that happens. DO NOT set things based on local
      // variables here.
    }

    // TODO: layout here?

    if ( this._boundsDirty ) {
      wasDirtyBefore = true;

      sceneryLog && sceneryLog.bounds && sceneryLog.bounds( 'bounds dirty' );

      // run this before firing the event
      this._boundsDirty = false;

      const oldBounds = scratchBounds2.set( ourBounds ); // store old value in a temporary Bounds2

      // no need to do the more expensive bounds transformation if we are still axis-aligned
      if ( this._transformBounds && !this._transform.getMatrix().isAxisAligned() ) {
        // mutates the matrix and bounds during recursion

        const matrix = scratchMatrix3.set( this.getMatrix() ); // calls below mutate this matrix
        ourBounds.set( Bounds2.NOTHING );
        // Include each painted self individually, transformed with the exact transform matrix.
        // This is expensive, as we have to do 2 matrix transforms for every descendant.
        this._includeTransformedSubtreeBounds( matrix, ourBounds ); // self and children

        const clipArea = this.clipArea;
        if ( clipArea ) {
          ourBounds.constrainBounds( clipArea.getBoundsWithTransform( matrix ) );
        }
      }
      else {
        // converts local to parent bounds. mutable methods used to minimize number of created bounds instances
        // (we create one so we don't change references to the old one)
        ourBounds.set( ourLocalBounds );
        this.transformBoundsFromLocalToParent( ourBounds );
      }

      sceneryLog && sceneryLog.bounds && sceneryLog.bounds( `bounds: ${ourBounds}` );

      if ( !ourBounds.equals( oldBounds ) ) {
        // if we have a bounds change, we need to invalidate our parents so they can be recomputed
        i = this._parents.length;
        while ( i-- ) {
          this._parents[ i ].invalidateBounds();
        }

        // TODO: consider changing to parameter object (that may be a problem for the GC overhead)
        if ( !ourBounds.equalsEpsilon( oldBounds, notificationThreshold ) ) {
          this.boundsProperty.notifyListeners( oldBounds ); // RE-ENTRANT CALL HERE, it will validateBounds()
        }
      }

      // WARNING: Think twice before adding code here below the listener notification. The notifyListeners() call can
      // trigger re-entrancy, so this function needs to work when that happens. DO NOT set things based on local
      // variables here.
    }

    // if there were side-effects, run the validation again until we are clean
    if ( this._childBoundsDirty || this._boundsDirty ) {
      sceneryLog && sceneryLog.bounds && sceneryLog.bounds( 'revalidation' );

      // TODO: if there are side-effects in listeners, this could overflow the stack. we should report an error
      // instead of locking up
      this.validateBounds(); // RE-ENTRANT CALL HERE, it will validateBounds()
    }

    if ( assert ) {
      assert( this._originalBounds === this.boundsProperty._value, 'Reference for bounds changed!' );
      assert( this._originalLocalBounds === this.localBoundsProperty._value, 'Reference for localBounds changed!' );
      assert( this._originalSelfBounds === this.selfBoundsProperty._value, 'Reference for selfBounds changed!' );
      assert( this._originalChildBounds === this.childBoundsProperty._value, 'Reference for childBounds changed!' );
    }

    // double-check that all of our bounds handling has been accurate
    if ( assertSlow ) {
      // new scope for safety
      ( () => {
        const epsilon = 0.000001;

        const childBounds = Bounds2.NOTHING.copy();
        _.each( this._children, child => {
          if ( !this._excludeInvisibleChildrenFromBounds || child.isVisible() ) {
            childBounds.includeBounds( child.boundsProperty._value );
          }
        } );

        let localBounds = this.selfBoundsProperty._value.union( childBounds );

        const clipArea = this.clipArea;
        if ( clipArea ) {
          localBounds = localBounds.intersection( clipArea.bounds );
        }

        const fullBounds = this.localToParentBounds( localBounds );

        assertSlow && assertSlow( this.childBoundsProperty._value.equalsEpsilon( childBounds, epsilon ),
          `Child bounds mismatch after validateBounds: ${
            this.childBoundsProperty._value.toString()}, expected: ${childBounds.toString()}` );
        assertSlow && assertSlow( this._localBoundsOverridden ||
                                  this._transformBounds ||
                                  this.boundsProperty._value.equalsEpsilon( fullBounds, epsilon ),
          `Bounds mismatch after validateBounds: ${this.boundsProperty._value.toString()
          }, expected: ${fullBounds.toString()}. This could have happened if a bounds instance owned by a Node` +
          ' was directly mutated (e.g. bounds.erode())' );
      } )();
    }

    sceneryLog && sceneryLog.bounds && sceneryLog.pop();

    return wasDirtyBefore; // whether any dirty flags were set
  }

  /**
   * Recursion for accurate transformed bounds handling. Mutates bounds with the added bounds.
   * Mutates the matrix (parameter), but mutates it back to the starting point (within floating-point error).
   */
  private _includeTransformedSubtreeBounds( matrix: Matrix3, bounds: Bounds2 ): Bounds2 {
    if ( !this.selfBounds.isEmpty() ) {
      bounds.includeBounds( this.getTransformedSelfBounds( matrix ) );
    }

    const numChildren = this._children.length;
    for ( let i = 0; i < numChildren; i++ ) {
      const child = this._children[ i ];

      matrix.multiplyMatrix( child._transform.getMatrix() );
      child._includeTransformedSubtreeBounds( matrix, bounds );
      matrix.multiplyMatrix( child._transform.getInverse() );
    }

    return bounds;
  }

  /**
   * Traverses this subtree and validates bounds only for subtrees that have bounds listeners (trying to exclude as
   * much as possible for performance). This is done so that we can do the minimum bounds validation to prevent any
   * bounds listeners from being triggered in further validateBounds() calls without other Node changes being done.
   * This is required for Display's atomic (non-reentrant) updateDisplay(), so that we don't accidentally trigger
   * bounds listeners while computing bounds during updateDisplay(). (scenery-internal)
   *
   * NOTE: this should pass by (ignore) any overridden localBounds, to trigger listeners below.
   */
  public validateWatchedBounds(): void {
    // Since a bounds listener on one of the roots could invalidate bounds on the other, we need to keep running this
    // until they are all clean. Otherwise, side-effects could occur from bounds validations
    // TODO: consider a way to prevent infinite loops here that occur due to bounds listeners triggering cycles
    while ( this.watchedBoundsScan() ) {
      // do nothing
    }
  }

  /**
   * Recursive function for validateWatchedBounds. Returned whether any validateBounds() returned true (means we have
   * to traverse again) - scenery-internal
   *
   * @returns - Whether there could have been any changes.
   */
  public watchedBoundsScan(): boolean {
    if ( this._boundsEventSelfCount !== 0 ) {
      // we are a root that should be validated. return whether we updated anything
      return this.validateBounds();
    }
    else if ( this._boundsEventCount > 0 && this._childBoundsDirty ) {
      // descendants have watched bounds, traverse!
      let changed = false;
      const numChildren = this._children.length;
      for ( let i = 0; i < numChildren; i++ ) {
        changed = this._children[ i ].watchedBoundsScan() || changed;
      }
      return changed;
    }
    else {
      // if _boundsEventCount is zero, no bounds are watched below us (don't traverse), and it wasn't changed
      return false;
    }
  }

  /**
   * Marks the bounds of this Node as invalid, so they are recomputed before being accessed again.
   */
  public invalidateBounds(): void {
    // TODO: sometimes we won't need to invalidate local bounds! it's not too much of a hassle though?
    this._boundsDirty = true;
    this._localBoundsDirty = true;

    // and set flags for all ancestors
    let i = this._parents.length;
    while ( i-- ) {
      this._parents[ i ].invalidateChildBounds();
    }
  }

  /**
   * Recursively tag all ancestors with _childBoundsDirty (scenery-internal)
   */
  public invalidateChildBounds(): void {
    // don't bother updating if we've already been tagged
    if ( !this._childBoundsDirty ) {
      this._childBoundsDirty = true;
      this._localBoundsDirty = true;
      let i = this._parents.length;
      while ( i-- ) {
        this._parents[ i ].invalidateChildBounds();
      }
    }
  }

  /**
   * Should be called to notify that our selfBounds needs to change to this new value.
   */
  public invalidateSelf( newSelfBounds?: Bounds2 ): void {
    assert && assert( newSelfBounds === undefined || newSelfBounds instanceof Bounds2,
      'invalidateSelf\'s newSelfBounds, if provided, needs to be Bounds2' );

    const ourSelfBounds = this.selfBoundsProperty._value;

    // If no self bounds are provided, rely on the bounds validation to trigger computation (using updateSelfBounds()).
    if ( !newSelfBounds ) {
      this._selfBoundsDirty = true;
      this.invalidateBounds();
      this._picker.onSelfBoundsDirty();
    }
    // Otherwise, set the self bounds directly
    else {
      assert && assert( newSelfBounds.isEmpty() || newSelfBounds.isFinite(), 'Bounds must be empty or finite in invalidateSelf' );

      // Don't recompute the self bounds
      this._selfBoundsDirty = false;

      // if these bounds are different than current self bounds
      if ( !ourSelfBounds.equals( newSelfBounds ) ) {
        const oldSelfBounds = scratchBounds2.set( ourSelfBounds );

        // set repaint flags
        this.invalidateBounds();
        this._picker.onSelfBoundsDirty();

        // record the new bounds
        ourSelfBounds.set( newSelfBounds );

        // fire the event immediately
        this.selfBoundsProperty.notifyListeners( oldSelfBounds );
      }
    }

    if ( assertSlow ) { this._picker.audit(); }
  }

  /**
   * Meant to be overridden by Node sub-types to compute self bounds (if invalidateSelf() with no arguments was called).
   *
   * @returns - Whether the self bounds changed.
   */
  protected updateSelfBounds(): boolean {
    // The Node implementation (un-overridden) will never change the self bounds (always NOTHING).
    assert && assert( this.selfBoundsProperty._value.equals( Bounds2.NOTHING ) );
    return false;
  }

  /**
   * Returns whether a Node is a child of this node.
   *
   * @returns - Whether potentialChild is actually our child.
   */
  public hasChild( potentialChild: Node ): boolean {
    assert && assert( potentialChild && ( potentialChild instanceof Node ), 'hasChild needs to be called with a Node' );
    const isOurChild = _.includes( this._children, potentialChild );
    assert && assert( isOurChild === _.includes( potentialChild._parents, this ), 'child-parent reference should match parent-child reference' );
    return isOurChild;
  }

  /**
   * Returns a Shape that represents the area covered by containsPointSelf.
   */
  public getSelfShape(): Shape {
    const selfBounds = this.selfBounds;
    if ( selfBounds.isEmpty() ) {
      return new Shape();
    }
    else {
      return Shape.bounds( this.selfBounds );
    }
  }

  /**
   * Returns our selfBounds (the bounds for this Node's content in the local coordinates, excluding anything from our
   * children and descendants).
   *
   * NOTE: Do NOT mutate the returned value!
   */
  public getSelfBounds(): Bounds2 {
    return this.selfBoundsProperty.value;
  }

  /**
   * See getSelfBounds() for more information
   */
  public get selfBounds(): Bounds2 {
    return this.getSelfBounds();
  }

  /**
   * Returns a bounding box that should contain all self content in the local coordinate frame (our normal self bounds
   * aren't guaranteed this for Text, etc.)
   *
   * Override this to provide different behavior.
   */
  public getSafeSelfBounds(): Bounds2 {
    return this.selfBoundsProperty.value;
  }

  /**
   * See getSafeSelfBounds() for more information
   */
  public get safeSelfBounds(): Bounds2 {
    return this.getSafeSelfBounds();
  }

  /**
   * Returns the bounding box that should contain all content of our children in our local coordinate frame. Does not
   * include our "self" bounds.
   *
   * NOTE: Do NOT mutate the returned value!
   */
  public getChildBounds(): Bounds2 {
    return this.childBoundsProperty.value;
  }

  /**
   * See getChildBounds() for more information
   */
  public get childBounds(): Bounds2 {
    return this.getChildBounds();
  }

  /**
   * Returns the bounding box that should contain all content of our children AND our self in our local coordinate
   * frame.
   *
   * NOTE: Do NOT mutate the returned value!
   */
  public getLocalBounds(): Bounds2 {
    return this.localBoundsProperty.value;
  }

  /**
   * See getLocalBounds() for more information
   */
  public get localBounds(): Bounds2 {
    return this.getLocalBounds();
  }

  /**
   * See setLocalBounds() for more information
   */
  public set localBounds( value: Bounds2 | null ) {
    this.setLocalBounds( value );
  }

  public get localBoundsOverridden(): boolean {
    return this._localBoundsOverridden;
  }

  /**
   * Allows overriding the value of localBounds (and thus changing things like 'bounds' that depend on localBounds).
   * If it's set to a non-null value, that value will always be used for localBounds until this function is called
   * again. To revert to having Scenery compute the localBounds, set this to null.  The bounds should not be reduced
   * smaller than the visible bounds on the screen.
   */
  public setLocalBounds( localBounds: Bounds2 | null ): this {
    assert && assert( localBounds === null || localBounds instanceof Bounds2, 'localBounds override should be set to either null or a Bounds2' );
    assert && assert( localBounds === null || !isNaN( localBounds.minX ), 'minX for localBounds should not be NaN' );
    assert && assert( localBounds === null || !isNaN( localBounds.minY ), 'minY for localBounds should not be NaN' );
    assert && assert( localBounds === null || !isNaN( localBounds.maxX ), 'maxX for localBounds should not be NaN' );
    assert && assert( localBounds === null || !isNaN( localBounds.maxY ), 'maxY for localBounds should not be NaN' );

    const ourLocalBounds = this.localBoundsProperty._value;
    const oldLocalBounds = ourLocalBounds.copy();

    if ( localBounds === null ) {
      // we can just ignore this if we weren't actually overriding local bounds before
      if ( this._localBoundsOverridden ) {

        this._localBoundsOverridden = false;
        this.localBoundsProperty.notifyListeners( oldLocalBounds );
        this.invalidateBounds();
      }
    }
    else {
      // just an instance check for now. consider equals() in the future depending on cost
      const changed = !localBounds.equals( ourLocalBounds ) || !this._localBoundsOverridden;

      if ( changed ) {
        ourLocalBounds.set( localBounds );
      }

      if ( !this._localBoundsOverridden ) {
        this._localBoundsOverridden = true; // NOTE: has to be done before invalidating bounds, since this disables localBounds computation
      }

      if ( changed ) {
        this.localBoundsProperty.notifyListeners( oldLocalBounds );
        this.invalidateBounds();
      }
    }

    return this; // allow chaining
  }

  /**
   * Meant to be overridden in sub-types that have more accurate bounds determination for when we are transformed.
   * Usually rotation is significant here, so that transformed bounds for non-rectangular shapes will be different.
   */
  public getTransformedSelfBounds( matrix: Matrix3 ): Bounds2 {
    // assume that we take up the entire rectangular bounds by default
    return this.selfBounds.transformed( matrix );
  }

  /**
   * Meant to be overridden in sub-types that have more accurate bounds determination for when we are transformed.
   * Usually rotation is significant here, so that transformed bounds for non-rectangular shapes will be different.
   *
   * This should include the "full" bounds that guarantee everything rendered should be inside (e.g. Text, where the
   * normal bounds may not be sufficient).
   */
  public getTransformedSafeSelfBounds( matrix: Matrix3 ): Bounds2 {
    return this.safeSelfBounds.transformed( matrix );
  }

  /**
   * Returns the visual "safe" bounds that are taken up by this Node and its subtree. Notably, this is essentially the
   * combined effects of the "visible" bounds (i.e. invisible nodes do not contribute to bounds), and "safe" bounds
   * (e.g. Text, where we need a larger bounds area to guarantee there is nothing outside). It also tries to "fit"
   * transformed bounds more tightly, where it will handle rotated Path bounds in an improved way.
   *
   * NOTE: This method is not optimized, and may create garbage and not be the fastest.
   *
   * @param [matrix] - If provided, will return the bounds assuming the content is transformed with the
   *                             given matrix.
   */
  public getSafeTransformedVisibleBounds( matrix?: Matrix3 ): Bounds2 {
    const localMatrix = ( matrix || Matrix3.IDENTITY ).timesMatrix( this.matrix );

    const bounds = Bounds2.NOTHING.copy();

    if ( this.visibleProperty.value ) {
      if ( !this.selfBounds.isEmpty() ) {
        bounds.includeBounds( this.getTransformedSafeSelfBounds( localMatrix ) );
      }

      if ( this._children.length ) {
        for ( let i = 0; i < this._children.length; i++ ) {
          bounds.includeBounds( this._children[ i ].getSafeTransformedVisibleBounds( localMatrix ) );
        }
      }
    }

    return bounds;
  }

  /**
   * See getSafeTransformedVisibleBounds() for more information -- This is called without any initial parameter
   */
  public get safeTransformedVisibleBounds(): Bounds2 {
    return this.getSafeTransformedVisibleBounds();
  }

  /**
   * Sets the flag that determines whether we will require more accurate (and expensive) bounds computation for this
   * node's transform.
   *
   * If set to false (default), Scenery will get the bounds of content, and then if rotated will determine the on-axis
   * bounds that completely cover the rotated bounds (potentially larger than actual content).
   * If set to true, Scenery will try to get the bounds of the actual rotated/transformed content.
   *
   * A good example of when this is necessary is if there are a bunch of nested children that each have pi/4 rotations.
   *
   * @param transformBounds - Whether accurate transform bounds should be used.
   */
  public setTransformBounds( transformBounds: boolean ): this {

    if ( this._transformBounds !== transformBounds ) {
      this._transformBounds = transformBounds;

      this.invalidateBounds();
    }

    return this; // allow chaining
  }

  /**
   * See setTransformBounds() for more information
   */
  public set transformBounds( value: boolean ) {
    this.setTransformBounds( value );
  }

  /**
   * See getTransformBounds() for more information
   */
  public get transformBounds(): boolean {
    return this.getTransformBounds();
  }

  /**
   * Returns whether accurate transformation bounds are used in bounds computation (see setTransformBounds).
   */
  public getTransformBounds(): boolean {
    return this._transformBounds;
  }

  /**
   * Returns the bounding box of this Node and all of its sub-trees (in the "parent" coordinate frame).
   *
   * NOTE: Do NOT mutate the returned value!
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getBounds(): Bounds2 {
    return this.boundsProperty.value;
  }

  /**
   * See getBounds() for more information
   */
  public get bounds(): Bounds2 {
    return this.getBounds();
  }

  /**
   * Like getLocalBounds() in the "local" coordinate frame, but includes only visible nodes.
   */
  public getVisibleLocalBounds(): Bounds2 {
    // defensive copy, since we use mutable modifications below
    const bounds = this.selfBounds.copy();

    let i = this._children.length;
    while ( i-- ) {
      bounds.includeBounds( this._children[ i ].getVisibleBounds() );
    }

    assert && assert( bounds.isFinite() || bounds.isEmpty(), 'Visible bounds should not be infinite' );
    return bounds;
  }

  /**
   * See getVisibleLocalBounds() for more information
   */
  public get visibleLocalBounds(): Bounds2 {
    return this.getVisibleLocalBounds();
  }

  /**
   * Like getBounds() in the "parent" coordinate frame, but includes only visible nodes
   */
  public getVisibleBounds(): Bounds2 {
    if ( this.isVisible() ) {
      return this.getVisibleLocalBounds().transform( this.getMatrix() );
    }
    else {
      return Bounds2.NOTHING;
    }
  }

  /**
   * See getVisibleBounds() for more information
   */
  public get visibleBounds(): Bounds2 {
    return this.getVisibleBounds();
  }

  /**
   * Tests whether the given point is "contained" in this node's subtree (optionally using mouse/touch areas), and if
   * so returns the Trail (rooted at this node) to the top-most (in stacking order) Node that contains the given
   * point.
   *
   * NOTE: This is optimized for the current input system (rather than what gets visually displayed on the screen), so
   * pickability (Node's pickable property, visibility, and the presence of input listeners) all may affect the
   * returned value.
   *
   * For example, hit-testing a simple shape (with no pickability) will return null:
   * > new phet.scenery.Circle( 20 ).hitTest( phet.dot.v2( 0, 0 ) ); // null
   *
   * If the same shape is made to be pickable, it will return a trail:
   * > new phet.scenery.Circle( 20, { pickable: true } ).hitTest( phet.dot.v2( 0, 0 ) );
   * > // returns a Trail with the circle as the only node.
   *
   * It will return the result that is visually stacked on top, so e.g.:
   * > new phet.scenery.Node( {
   * >   pickable: true,
   * >   children: [
   * >     new phet.scenery.Circle( 20 ),
   * >     new phet.scenery.Circle( 15 )
   * >   ]
   * > } ).hitTest( phet.dot.v2( 0, 0 ) ); // returns the "top-most" circle (the one with radius:15).
   *
   * This is used by Scenery's internal input system by calling hitTest on a Display's rootNode with the
   * global-coordinate point.
   *
   * @param point - The point (in the parent coordinate frame) to check against this node's subtree.
   * @param [isMouse] - Whether mouseAreas should be used.
   * @param [isTouch] - Whether touchAreas should be used.
   * @returns - Returns null if the point is not contained in the subtree.
   */
  public hitTest( point: Vector2, isMouse?: boolean, isTouch?: boolean ): Trail | null {
    assert && assert( point.isFinite(), 'The point should be a finite Vector2' );
    assert && assert( isMouse === undefined || typeof isMouse === 'boolean',
      'If isMouse is provided, it should be a boolean' );
    assert && assert( isTouch === undefined || typeof isTouch === 'boolean',
      'If isTouch is provided, it should be a boolean' );

    return this._picker.hitTest( point, !!isMouse, !!isTouch );
  }

  /**
   * Hit-tests what is under the pointer, and returns a {Trail} to that Node (or null if there is no matching node).
   *
   * See hitTest() for more details about what will be returned.
   */
  public trailUnderPointer( pointer: Pointer ): Trail | null {
    return pointer.point === null ? null : this.hitTest( pointer.point, pointer instanceof Mouse, pointer.isTouchLike() );
  }

  /**
   * Returns whether a point (in parent coordinates) is contained in this node's sub-tree.
   *
   * See hitTest() for more details about what will be returned.
   *
   * @returns - Whether the point is contained.
   */
  public containsPoint( point: Vector2 ): boolean {
    return this.hitTest( point ) !== null;
  }

  /**
   * Override this for computation of whether a point is inside our self content (defaults to selfBounds check).
   *
   * @param point - Considered to be in the local coordinate frame
   */
  public containsPointSelf( point: Vector2 ): boolean {
    // if self bounds are not null default to checking self bounds
    return this.selfBounds.containsPoint( point );
  }

  /**
   * Returns whether this node's selfBounds is intersected by the specified bounds.
   *
   * @param bounds - Bounds to test, assumed to be in the local coordinate frame.
   */
  public intersectsBoundsSelf( bounds: Bounds2 ): boolean {
    // if self bounds are not null, child should override this
    return this.selfBounds.intersectsBounds( bounds );
  }

  public isPhetioMouseHittable( point: Vector2 ): boolean {

    // invisible things cannot be autoselected
    if ( !this.visible ) {
      return false;
    }

    // unpickable things cannot be autoselected
    if ( this.pickable === false ) {
      return false;
    }

    // Transform the point in the local coordinate frame, so we can test it with the clipArea/children
    const localPoint = this._transform.getInverse().timesVector2( point );

    // If our point is outside of the local-coordinate clipping area, there should be no hit.
    if ( this.clipArea !== null && !this.clipArea.containsPoint( localPoint ) ) {
      return false;
    }
    return true;
  }

  // Used in Studio Autoselect.  Returns an instrumented PhET-iO Element Node if possible.
  // Adapted from Picker.recursiveHitTest
  // @returns - may not be a Node.  For instance, ThreeIsometricNode hits Mass instances
  public getPhetioMouseHit( point: Vector2 ): PhetioObject | null {

    if ( !this.isPhetioMouseHittable( point ) ) {
      return null;
    }

    // Transform the point in the local coordinate frame, so we can test it with the clipArea/children
    const localPoint = this._transform.getInverse().timesVector2( point );

    // Check children before our "self", since the children are rendered on top.
    // Manual iteration here so we can return directly, and so we can iterate backwards (last node is in front).
    for ( let i = this._children.length - 1; i >= 0; i-- ) {
      const child = this._children[ i ];
      const childHit = child.getPhetioMouseHit( localPoint );

      // If there was a hit, immediately add our node to the start of the Trail (will recursively build the Trail).
      if ( childHit ) {
        return childHit.isPhetioInstrumented() ? childHit : this;
      }
    }

    // Tests for mouse and touch hit areas before testing containsPointSelf
    if ( this._mouseArea ) {
      // NOTE: both Bounds2 and Shape have containsPoint! We use both here!
      return this._mouseArea.containsPoint( localPoint ) ? this : null;
    }

    // Didn't hit our children, so check ourself as a last resort. Check our selfBounds first, so we can potentially
    // avoid hit-testing the actual object (which may be more expensive).
    if ( this.selfBounds.containsPoint( localPoint ) ) {
      if ( this.containsPointSelf( localPoint ) ) {
        return this;
      }
    }

    // No hit
    return null;
  }

  /**
   * Whether this Node itself is painted (displays something itself). Meant to be overridden.
   */
  public isPainted(): boolean {
    // Normal nodes don't render anything
    return false;
  }

  /**
   * Whether this Node's selfBounds are considered to be valid (always containing the displayed self content
   * of this node). Meant to be overridden in subtypes when this can change (e.g. Text).
   *
   * If this value would potentially change, please trigger the event 'selfBoundsValid'.
   */
  public areSelfBoundsValid(): boolean {
    return true;
  }

  /**
   * Returns whether this Node has any parents at all.
   */
  public hasParent(): boolean {
    return this._parents.length !== 0;
  }

  /**
   * Returns whether this Node has any children at all.
   */
  public hasChildren(): boolean {
    return this._children.length > 0;
  }

  /**
   * Returns whether a child should be included for layout (if this Node is a layout container).
   */
  public isChildIncludedInLayout( child: Node ): boolean {
    return child.bounds.isValid() && ( !this._excludeInvisibleChildrenFromBounds || child.visible );
  }

  /**
   * Calls the callback on nodes recursively in a depth-first manner.
   */
  public walkDepthFirst( callback: ( node: Node ) => void ): void {
    callback( this );
    const length = this._children.length;
    for ( let i = 0; i < length; i++ ) {
      this._children[ i ].walkDepthFirst( callback );
    }
  }

  /**
   * Adds an input listener.
   *
   * See Input.js documentation for information about how event listeners are used.
   *
   * Additionally, the following fields are supported on a listener:
   *
   * - interrupt {function()}: When a pointer is interrupted, it will attempt to call this method on the input listener
   * - cursor {string|null}: If node.cursor is null, any non-null cursor of an input listener will effectively
   *                         "override" it. NOTE: this can be implemented as an es5 getter, if the cursor can change
   */
  public addInputListener( listener: TInputListener ): this {
    assert && assert( !_.includes( this._inputListeners, listener ), 'Input listener already registered on this Node' );
    assert && assert( listener !== null, 'Input listener cannot be null' );
    assert && assert( listener !== undefined, 'Input listener cannot be undefined' );

    // don't allow listeners to be added multiple times
    if ( !_.includes( this._inputListeners, listener ) ) {
      this._inputListeners.push( listener );
      this._picker.onAddInputListener();
      if ( assertSlow ) { this._picker.audit(); }
    }
    return this;
  }

  /**
   * Removes an input listener that was previously added with addInputListener.
   */
  public removeInputListener( listener: TInputListener ): this {
    const index = _.indexOf( this._inputListeners, listener );

    // ensure the listener is in our list (ignore assertion for disposal, see https://github.com/phetsims/sun/issues/394)
    assert && assert( this.isDisposed || index >= 0, 'Could not find input listener to remove' );
    if ( index >= 0 ) {
      this._inputListeners.splice( index, 1 );
      this._picker.onRemoveInputListener();
      if ( assertSlow ) { this._picker.audit(); }
    }

    return this;
  }

  /**
   * Returns whether this input listener is currently listening to this node.
   *
   * More efficient than checking node.inputListeners, as that includes a defensive copy.
   */
  public hasInputListener( listener: TInputListener ): boolean {
    for ( let i = 0; i < this._inputListeners.length; i++ ) {
      if ( this._inputListeners[ i ] === listener ) {
        return true;
      }
    }
    return false;
  }

  /**
   * Interrupts all input listeners that are attached to this node.
   */
  public interruptInput(): this {
    const listenersCopy = this.inputListeners;

    for ( let i = 0; i < listenersCopy.length; i++ ) {
      const listener = listenersCopy[ i ];

      listener.interrupt && listener.interrupt();
    }

    return this;
  }

  /**
   * Interrupts all input listeners that are attached to either this node, or a descendant node.
   */
  public interruptSubtreeInput(): this {
    this.interruptInput();

    const children = this._children.slice();
    for ( let i = 0; i < children.length; i++ ) {
      children[ i ].interruptSubtreeInput();
    }

    return this;
  }

  /**
   * Changes the transform of this Node by adding a transform. The default "appends" the transform, so that it will
   * appear to happen to the Node before the rest of the transform would apply, but if "prepended", the rest of the
   * transform would apply first.
   *
   * As an example, if a Node is centered at (0,0) and scaled by 2:
   * translate( 100, 0 ) would cause the center of the Node (in the parent coordinate frame) to be at (200,0).
   * translate( 100, 0, true ) would cause the center of the Node (in the parent coordinate frame) to be at (100,0).
   *
   * Allowed call signatures:
   * translate( x {number}, y {number} )
   * translate( x {number}, y {number}, prependInstead {boolean} )
   * translate( vector {Vector2} )
   * translate( vector {Vector2}, prependInstead {boolean} )
   *
   * @param x - The x coordinate
   * @param y - The y coordinate
   * @param [prependInstead] - Whether the transform should be prepended (defaults to false)
   */
  public translate( v: Vector2, prependInstead?: boolean ): void;
  translate( x: number, y: number, prependInstead?: boolean ): void; // eslint-disable-line @typescript-eslint/explicit-member-accessibility
  translate( x: number | Vector2, y?: number | boolean, prependInstead?: boolean ): void { // eslint-disable-line @typescript-eslint/explicit-member-accessibility
    if ( typeof x === 'number' ) {
      // translate( x, y, prependInstead )
      assert && assert( isFinite( x ), 'x should be a finite number' );

      assert && assert( typeof y === 'number' && isFinite( y ), 'y should be a finite number' ); // eslint-disable-line no-simple-type-checking-assertions

      if ( Math.abs( x ) < 1e-12 && Math.abs( y as number ) < 1e-12 ) { return; } // bail out if both are zero
      if ( prependInstead ) {
        this.prependTranslation( x, y as number );
      }
      else {
        this.appendMatrix( scratchMatrix3.setToTranslation( x, y as number ) );
      }
    }
    else {
      // translate( vector, prependInstead )
      const vector = x;
      assert && assert( vector.isFinite(), 'translation should be a finite Vector2 if not finite numbers' );
      if ( !vector.x && !vector.y ) { return; } // bail out if both are zero
      this.translate( vector.x, vector.y, y as boolean ); // forward to full version
    }
  }

  /**
   * Scales the node's transform. The default "appends" the transform, so that it will
   * appear to happen to the Node before the rest of the transform would apply, but if "prepended", the rest of the
   * transform would apply first.
   *
   * As an example, if a Node is translated to (100,0):
   * scale( 2 ) will leave the Node translated at (100,0), but it will be twice as big around its origin at that location.
   * scale( 2, true ) will shift the Node to (200,0).
   *
   * Allowed call signatures:
   * (s invocation): scale( s {number|Vector2}, [prependInstead] {boolean} )
   * (x,y invocation): scale( x {number}, y {number}, [prependInstead] {boolean} )
   *
   * @param x - (s invocation): {number} scales both dimensions equally, or {Vector2} scales independently
   *          - (x,y invocation): {number} scale for the x-dimension
   * @param [y] - (s invocation): {boolean} prependInstead - Whether the transform should be prepended (defaults to false)
   *            - (x,y invocation): {number} y - scale for the y-dimension
   * @param [prependInstead] - (x,y invocation) Whether the transform should be prepended (defaults to false)
   */
  public scale( s: number, prependInstead?: boolean ): void;
  scale( s: Vector2, prependInstead?: boolean ): void; // eslint-disable-line @typescript-eslint/explicit-member-accessibility
  scale( x: number, y: number, prependInstead?: boolean ): void; // eslint-disable-line @typescript-eslint/explicit-member-accessibility
  scale( x: number | Vector2, y?: number | boolean, prependInstead?: boolean ): void { // eslint-disable-line @typescript-eslint/explicit-member-accessibility
    if ( typeof x === 'number' ) {
      assert && assert( isFinite( x ), 'scales should be finite' );
      if ( y === undefined || typeof y === 'boolean' ) {
        // scale( scale, [prependInstead] )
        this.scale( x, x, y );
      }
      else {
        // scale( x, y, [prependInstead] )
        assert && assert( isFinite( y ), 'scales should be finite numbers' );
        assert && assert( prependInstead === undefined || typeof prependInstead === 'boolean', 'If provided, prependInstead should be boolean' );

        if ( x === 1 && y === 1 ) { return; } // bail out if we are scaling by 1 (identity)
        if ( prependInstead ) {
          this.prependMatrix( Matrix3.scaling( x, y ) );
        }
        else {
          this.appendMatrix( Matrix3.scaling( x, y ) );
        }
      }
    }
    else {
      // scale( vector, [prependInstead] )
      const vector = x;
      assert && assert( vector.isFinite(), 'scale should be a finite Vector2 if not a finite number' );
      this.scale( vector.x, vector.y, y as boolean ); // forward to full version
    }
  }

  /**
   * Rotates the node's transform. The default "appends" the transform, so that it will
   * appear to happen to the Node before the rest of the transform would apply, but if "prepended", the rest of the
   * transform would apply first.
   *
   * As an example, if a Node is translated to (100,0):
   * rotate( Math.PI ) will rotate the Node around (100,0)
   * rotate( Math.PI, true ) will rotate the Node around the origin, moving it to (-100,0)
   *
   * @param angle - The angle (in radians) to rotate by
   * @param [prependInstead] - Whether the transform should be prepended (defaults to false)
   */
  public rotate( angle: number, prependInstead?: boolean ): void {
    assert && assert( isFinite( angle ), 'angle should be a finite number' );
    assert && assert( prependInstead === undefined || typeof prependInstead === 'boolean' );
    if ( angle % ( 2 * Math.PI ) === 0 ) { return; } // bail out if our angle is effectively 0
    if ( prependInstead ) {
      this.prependMatrix( Matrix3.rotation2( angle ) );
    }
    else {
      this.appendMatrix( Matrix3.rotation2( angle ) );
    }
  }

  /**
   * Rotates the node's transform around a specific point (in the parent coordinate frame) by prepending the transform.
   *
   * TODO: determine whether this should use the appendMatrix method
   *
   * @param point - In the parent coordinate frame
   * @param angle - In radians
   */
  public rotateAround( point: Vector2, angle: number ): this {
    assert && assert( point.isFinite(), 'point should be a finite Vector2' );
    assert && assert( isFinite( angle ), 'angle should be a finite number' );

    let matrix = Matrix3.translation( -point.x, -point.y );
    matrix = Matrix3.rotation2( angle ).timesMatrix( matrix );
    matrix = Matrix3.translation( point.x, point.y ).timesMatrix( matrix );
    this.prependMatrix( matrix );
    return this;
  }

  /**
   * Shifts the x coordinate (in the parent coordinate frame) of where the node's origin is transformed to.
   */
  public setX( x: number ): this {
    assert && assert( isFinite( x ), 'x should be a finite number' );

    this.translate( x - this.getX(), 0, true );
    return this;
  }

  /**
   * See setX() for more information
   */
  public set x( value: number ) {
    this.setX( value );
  }

  /**
   * See getX() for more information
   */
  public get x(): number {
    return this.getX();
  }

  /**
   * Returns the x coordinate (in the parent coordinate frame) of where the node's origin is transformed to.
   */
  public getX(): number {
    return this._transform.getMatrix().m02();
  }

  /**
   * Shifts the y coordinate (in the parent coordinate frame) of where the node's origin is transformed to.
   */
  public setY( y: number ): this {
    assert && assert( isFinite( y ), 'y should be a finite number' );

    this.translate( 0, y - this.getY(), true );
    return this;
  }

  /**
   * See setY() for more information
   */
  public set y( value: number ) {
    this.setY( value );
  }

  /**
   * See getY() for more information
   */
  public get y(): number {
    return this.getY();
  }

  /**
   * Returns the y coordinate (in the parent coordinate frame) of where the node's origin is transformed to.
   */
  public getY(): number {
    return this._transform.getMatrix().m12();
  }

  /**
   * Typically without rotations or negative parameters, this sets the scale for each axis. In its more general form,
   * it modifies the node's transform so that:
   * - Transforming (1,0) with our transform will result in a vector with magnitude abs( x-scale-magnitude )
   * - Transforming (0,1) with our transform will result in a vector with magnitude abs( y-scale-magnitude )
   * - If parameters are negative, it will flip orientation in that direct.
   *
   * Allowed call signatures:
   * setScaleMagnitude( s )
   * setScaleMagnitude( sx, sy )
   * setScaleMagnitude( vector )
   *
   * @param a - Scale for both axes, or scale for x-axis if using the 2-parameter call
   * @param [b] - Scale for the Y axis (only for the 2-parameter call)
   */
  public setScaleMagnitude( s: number ): this;
  setScaleMagnitude( v: Vector2 ): this; // eslint-disable-line @typescript-eslint/explicit-member-accessibility
  setScaleMagnitude( sx: number, sy: number ): this; // eslint-disable-line @typescript-eslint/explicit-member-accessibility
  setScaleMagnitude( a: number | Vector2, b?: number ): this { // eslint-disable-line @typescript-eslint/explicit-member-accessibility
    const currentScale = this.getScaleVector();

    if ( typeof a === 'number' ) {
      if ( b === undefined ) {
        // to map setScaleMagnitude( scale ) => setScaleMagnitude( scale, scale )
        b = a;
      }
      assert && assert( isFinite( a ), 'setScaleMagnitude parameters should be finite numbers' );
      assert && assert( isFinite( b ), 'setScaleMagnitude parameters should be finite numbers' );
      // setScaleMagnitude( x, y )
      this.appendMatrix( Matrix3.scaling( a / currentScale.x, b / currentScale.y ) );
    }
    else {
      // setScaleMagnitude( vector ), where we set the x-scale to vector.x and y-scale to vector.y
      assert && assert( a.isFinite(), 'first parameter should be a finite Vector2' );

      this.appendMatrix( Matrix3.scaling( a.x / currentScale.x, a.y / currentScale.y ) );
    }
    return this;
  }

  /**
   * Returns a vector with an entry for each axis, e.g. (5,2) for an affine matrix with rows ((5,0,0),(0,2,0),(0,0,1)).
   *
   * It is equivalent to:
   * ( T(1,0).magnitude(), T(0,1).magnitude() ) where T() transforms points with our transform.
   */
  public getScaleVector(): Vector2 {
    return this._transform.getMatrix().getScaleVector();
  }

  /**
   * Rotates this node's transform so that a unit (1,0) vector would be rotated by this node's transform by the
   * specified amount.
   *
   * @param rotation - In radians
   */
  public setRotation( rotation: number ): this {
    assert && assert( isFinite( rotation ),
      'rotation should be a finite number' );

    this.appendMatrix( scratchMatrix3.setToRotationZ( rotation - this.getRotation() ) );
    return this;
  }

  /**
   * See setRotation() for more information
   */
  public set rotation( value: number ) {
    this.setRotation( value );
  }

  /**
   * See getRotation() for more information
   */
  public get rotation(): number {
    return this.getRotation();
  }

  /**
   * Returns the rotation (in radians) that would be applied to a unit (1,0) vector when transformed with this Node's
   * transform.
   */
  public getRotation(): number {
    return this._transform.getMatrix().getRotation();
  }

  /**
   * Modifies the translation of this Node's transform so that the node's local-coordinate origin will be transformed
   * to the passed-in x/y.
   *
   * Allowed call signatures:
   * setTranslation( x, y )
   * setTranslation( vector )
   *
   * @param a - X translation - or Vector with x/y translation in components
   * @param [b] - Y translation
   */
  public setTranslation( x: number, y: number ): this;
  setTranslation( v: Vector2 ): this; // eslint-disable-line @typescript-eslint/explicit-member-accessibility
  setTranslation( a: number | Vector2, b?: number ): this { // eslint-disable-line @typescript-eslint/explicit-member-accessibility
    const m = this._transform.getMatrix();
    const tx = m.m02();
    const ty = m.m12();

    let dx;
    let dy;

    if ( typeof a === 'number' ) {
      assert && assert( isFinite( a ), 'Parameters to setTranslation should be finite numbers' );
      assert && assert( b !== undefined && isFinite( b ), 'Parameters to setTranslation should be finite numbers' );
      dx = a - tx;
      dy = b! - ty;
    }
    else {
      assert && assert( a.isFinite(), 'Should be a finite Vector2' );
      dx = a.x - tx;
      dy = a.y - ty;
    }

    this.translate( dx, dy, true );

    return this;
  }

  /**
   * See setTranslation() for more information - this should only be used with Vector2
   */
  public set translation( value: Vector2 ) {
    this.setTranslation( value );
  }

  /**
   * See getTranslation() for more information
   */
  public get translation(): Vector2 {
    return this.getTranslation();
  }

  /**
   * Returns a vector of where this Node's local-coordinate origin will be transformed by it's own transform.
   */
  public getTranslation(): Vector2 {
    const matrix = this._transform.getMatrix();
    return new Vector2( matrix.m02(), matrix.m12() );
  }

  /**
   * Appends a transformation matrix to this Node's transform. Appending means this transform is conceptually applied
   * first before the rest of the Node's current transform (i.e. applied in the local coordinate frame).
   */
  public appendMatrix( matrix: Matrix3 ): void {
    assert && assert( matrix.isFinite(), 'matrix should be a finite Matrix3' );
    assert && assert( matrix.getDeterminant() !== 0, 'matrix should not map plane to a line or point' );
    this._transform.append( matrix );
  }

  /**
   * Prepends a transformation matrix to this Node's transform. Prepending means this transform is conceptually applied
   * after the rest of the Node's current transform (i.e. applied in the parent coordinate frame).
   */
  public prependMatrix( matrix: Matrix3 ): void {
    assert && assert( matrix.isFinite(), 'matrix should be a finite Matrix3' );
    assert && assert( matrix.getDeterminant() !== 0, 'matrix should not map plane to a line or point' );
    this._transform.prepend( matrix );
  }

  /**
   * Prepends an (x,y) translation to our Node's transform in an efficient manner without allocating a matrix.
   * see https://github.com/phetsims/scenery/issues/119
   */
  public prependTranslation( x: number, y: number ): void {
    assert && assert( isFinite( x ), 'x should be a finite number' );
    assert && assert( isFinite( y ), 'y should be a finite number' );

    if ( !x && !y ) { return; } // bail out if both are zero

    this._transform.prependTranslation( x, y );
  }

  /**
   * Changes this Node's transform to match the passed-in transformation matrix.
   */
  public setMatrix( matrix: Matrix3 ): void {
    assert && assert( matrix.isFinite(), 'matrix should be a finite Matrix3' );
    assert && assert( matrix.getDeterminant() !== 0, 'matrix should not map plane to a line or point' );

    this._transform.setMatrix( matrix );
  }

  /**
   * See setMatrix() for more information
   */
  public set matrix( value: Matrix3 ) {
    this.setMatrix( value );
  }

  /**
   * See getMatrix() for more information
   */
  public get matrix(): Matrix3 {
    return this.getMatrix();
  }

  /**
   * Returns a Matrix3 representing our Node's transform.
   *
   * NOTE: Do not mutate the returned matrix.
   */
  public getMatrix(): Matrix3 {
    return this._transform.getMatrix();
  }

  /**
   * Returns a reference to our Node's transform
   */
  public getTransform(): Transform3 {
    // for now, return an actual copy. we can consider listening to changes in the future
    return this._transform;
  }

  /**
   * See getTransform() for more information
   */
  public get transform(): Transform3 {
    return this.getTransform();
  }

  /**
   * Resets our Node's transform to an identity transform (i.e. no transform is applied).
   */
  public resetTransform(): void {
    this.setMatrix( Matrix3.IDENTITY );
  }

  /**
   * Callback function that should be called when our transform is changed.
   */
  private onTransformChange(): void {
    // TODO: why is local bounds invalidation needed here?
    this.invalidateBounds();

    this._picker.onTransformChange();
    if ( assertSlow ) { this._picker.audit(); }

    this.transformEmitter.emit();
  }

  /**
   * Called when our summary bitmask changes (scenery-internal)
   */
  public onSummaryChange( oldBitmask: number, newBitmask: number ): void {
    // Defined in ParallelDOM.js
    this._pdomDisplaysInfo.onSummaryChange( oldBitmask, newBitmask );
  }

  /**
   * Updates our node's scale and applied scale factor if we need to change our scale to fit within the maximum
   * dimensions (maxWidth and maxHeight). See documentation in constructor for detailed behavior.
   */
  private updateMaxDimension( localBounds: Bounds2 ): void {
    assert && this.auditMaxDimensions();

    const currentScale = this._appliedScaleFactor;
    let idealScale = 1;

    if ( this._maxWidth !== null ) {
      const width = localBounds.width;
      if ( width > this._maxWidth ) {
        idealScale = Math.min( idealScale, this._maxWidth / width );
      }
    }

    if ( this._maxHeight !== null ) {
      const height = localBounds.height;
      if ( height > this._maxHeight ) {
        idealScale = Math.min( idealScale, this._maxHeight / height );
      }
    }

    const scaleAdjustment = idealScale / currentScale;
    if ( scaleAdjustment !== 1 ) {
      // Set this first, for supporting re-entrancy if our content changes based on the scale
      this._appliedScaleFactor = idealScale;

      this.scale( scaleAdjustment );
    }
  }

  /**
   * Scenery-internal method for verifying maximum dimensions are NOT smaller than preferred dimensions
   * NOTE: This has to be public due to mixins not able to access protected/private methods
   */
  public auditMaxDimensions(): void {
    assert && assert( this._maxWidth === null || !isWidthSizable( this ) || this.preferredWidth === null || this._maxWidth >= this.preferredWidth - 1e-7,
      'If maxWidth and preferredWidth are both non-null, maxWidth should NOT be smaller than the preferredWidth. If that happens, it would trigger an infinite loop' );

    assert && assert( this._maxHeight === null || !isHeightSizable( this ) || this.preferredHeight === null || this._maxHeight >= this.preferredHeight - 1e-7,
      'If maxHeight and preferredHeight are both non-null, maxHeight should NOT be smaller than the preferredHeight. If that happens, it would trigger an infinite loop' );
  }

  /**
   * Increments/decrements bounds "listener" count based on the values of maxWidth/maxHeight before and after.
   * null is like no listener, non-null is like having a listener, so we increment for null => non-null, and
   * decrement for non-null => null.
   */
  private onMaxDimensionChange( beforeMaxLength: number | null, afterMaxLength: number | null ): void {
    if ( beforeMaxLength === null && afterMaxLength !== null ) {
      this.changeBoundsEventCount( 1 );
      this._boundsEventSelfCount++;
    }
    else if ( beforeMaxLength !== null && afterMaxLength === null ) {
      this.changeBoundsEventCount( -1 );
      this._boundsEventSelfCount--;
    }
  }

  /**
   * Sets the maximum width of the Node (see constructor for documentation on how maximum dimensions work).
   */
  public setMaxWidth( maxWidth: number | null ): void {
    assert && assert( maxWidth === null || ( typeof maxWidth === 'number' && maxWidth > 0 ),
      'maxWidth should be null (no constraint) or a positive number' );

    if ( this._maxWidth !== maxWidth ) {
      // update synthetic bounds listener count (to ensure our bounds are validated at the start of updateDisplay)
      this.onMaxDimensionChange( this._maxWidth, maxWidth );

      this._maxWidth = maxWidth;

      this.updateMaxDimension( this.localBoundsProperty.value );
    }
  }

  /**
   * See setMaxWidth() for more information
   */
  public set maxWidth( value: number | null ) {
    this.setMaxWidth( value );
  }

  /**
   * See getMaxWidth() for more information
   */
  public get maxWidth(): number | null {
    return this.getMaxWidth();
  }

  /**
   * Returns the maximum width (if any) of the Node.
   */
  public getMaxWidth(): number | null {
    return this._maxWidth;
  }

  /**
   * Sets the maximum height of the Node (see constructor for documentation on how maximum dimensions work).
   */
  public setMaxHeight( maxHeight: number | null ): void {
    assert && assert( maxHeight === null || ( typeof maxHeight === 'number' && maxHeight > 0 ),
      'maxHeight should be null (no constraint) or a positive number' );

    if ( this._maxHeight !== maxHeight ) {
      // update synthetic bounds listener count (to ensure our bounds are validated at the start of updateDisplay)
      this.onMaxDimensionChange( this._maxHeight, maxHeight );

      this._maxHeight = maxHeight;

      this.updateMaxDimension( this.localBoundsProperty.value );
    }
  }

  /**
   * See setMaxHeight() for more information
   */
  public set maxHeight( value: number | null ) {
    this.setMaxHeight( value );
  }

  /**
   * See getMaxHeight() for more information
   */
  public get maxHeight(): number | null {
    return this.getMaxHeight();
  }

  /**
   * Returns the maximum height (if any) of the Node.
   */
  public getMaxHeight(): number | null {
    return this._maxHeight;
  }

  /**
   * Shifts this Node horizontally so that its left bound (in the parent coordinate frame) is equal to the passed-in
   * 'left' X value.
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param left - After this operation, node.left should approximately equal this value.
   */
  public setLeft( left: number ): this {
    const currentLeft = this.getLeft();
    if ( isFinite( currentLeft ) ) {
      this.translate( left - currentLeft, 0, true );
    }

    return this; // allow chaining
  }

  /**
   * See setLeft() for more information
   */
  public set left( value: number ) {
    this.setLeft( value );
  }

  /**
   * See getLeft() for more information
   */
  public get left(): number {
    return this.getLeft();
  }

  /**
   * Returns the X value of the left side of the bounding box of this Node (in the parent coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getLeft(): number {
    return this.getBounds().minX;
  }

  /**
   * Shifts this Node horizontally so that its right bound (in the parent coordinate frame) is equal to the passed-in
   * 'right' X value.
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param right - After this operation, node.right should approximately equal this value.
   */
  public setRight( right: number ): this {
    const currentRight = this.getRight();
    if ( isFinite( currentRight ) ) {
      this.translate( right - currentRight, 0, true );
    }
    return this; // allow chaining
  }

  /**
   * See setRight() for more information
   */
  public set right( value: number ) {
    this.setRight( value );
  }

  /**
   * See getRight() for more information
   */
  public get right(): number {
    return this.getRight();
  }

  /**
   * Returns the X value of the right side of the bounding box of this Node (in the parent coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getRight(): number {
    return this.getBounds().maxX;
  }

  /**
   * Shifts this Node horizontally so that its horizontal center (in the parent coordinate frame) is equal to the
   * passed-in center X value.
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param x - After this operation, node.centerX should approximately equal this value.
   */
  public setCenterX( x: number ): this {
    const currentCenterX = this.getCenterX();
    if ( isFinite( currentCenterX ) ) {
      this.translate( x - currentCenterX, 0, true );
    }

    return this; // allow chaining
  }

  /**
   * See setCenterX() for more information
   */
  public set centerX( value: number ) {
    this.setCenterX( value );
  }

  /**
   * See getCenterX() for more information
   */
  public get centerX(): number {
    return this.getCenterX();
  }

  /**
   * Returns the X value of this node's horizontal center (in the parent coordinate frame)
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getCenterX(): number {
    return this.getBounds().getCenterX();
  }

  /**
   * Shifts this Node vertically so that its vertical center (in the parent coordinate frame) is equal to the
   * passed-in center Y value.
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param y - After this operation, node.centerY should approximately equal this value.
   */
  public setCenterY( y: number ): this {
    const currentCenterY = this.getCenterY();
    if ( isFinite( currentCenterY ) ) {
      this.translate( 0, y - currentCenterY, true );
    }

    return this; // allow chaining
  }

  /**
   * See setCenterY() for more information
   */
  public set centerY( value: number ) {
    this.setCenterY( value );
  }

  /**
   * See getCenterX() for more information
   */
  public get centerY(): number {
    return this.getCenterY();
  }

  /**
   * Returns the Y value of this node's vertical center (in the parent coordinate frame)
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getCenterY(): number {
    return this.getBounds().getCenterY();
  }

  /**
   * Shifts this Node vertically so that its top (in the parent coordinate frame) is equal to the passed-in Y value.
   *
   * NOTE: top is the lowest Y value in our bounds.
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param top - After this operation, node.top should approximately equal this value.
   */
  public setTop( top: number ): this {
    const currentTop = this.getTop();
    if ( isFinite( currentTop ) ) {
      this.translate( 0, top - currentTop, true );
    }

    return this; // allow chaining
  }

  /**
   * See setTop() for more information
   */
  public set top( value: number ) {
    this.setTop( value );
  }

  /**
   * See getTop() for more information
   */
  public get top(): number {
    return this.getTop();
  }

  /**
   * Returns the lowest Y value of this node's bounding box (in the parent coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getTop(): number {
    return this.getBounds().minY;
  }

  /**
   * Shifts this Node vertically so that its bottom (in the parent coordinate frame) is equal to the passed-in Y value.
   *
   * NOTE: bottom is the highest Y value in our bounds.
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   *
   * @param bottom - After this operation, node.bottom should approximately equal this value.
   */
  public setBottom( bottom: number ): this {
    const currentBottom = this.getBottom();
    if ( isFinite( currentBottom ) ) {
      this.translate( 0, bottom - currentBottom, true );
    }

    return this; // allow chaining
  }

  /**
   * See setBottom() for more information
   */
  public set bottom( value: number ) {
    this.setBottom( value );
  }

  /**
   * See getBottom() for more information
   */
  public get bottom(): number {
    return this.getBottom();
  }

  /**
   * Returns the highest Y value of this node's bounding box (in the parent coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getBottom(): number {
    return this.getBounds().maxY;
  }

  /*
   * Convenience locations
   *
   * Upper is in terms of the visual layout in Scenery and other programs, so the minY is the "upper", and minY is the "lower"
   *
   *             left (x)     centerX        right
   *          ---------------------------------------
   * top  (y) | leftTop     centerTop     rightTop
   * centerY  | leftCenter  center        rightCenter
   * bottom   | leftBottom  centerBottom  rightBottom
   *
   * NOTE: This requires computation of this node's subtree bounds, which may incur some performance loss.
   */

  /**
   * Sets the position of the upper-left corner of this node's bounds to the specified point.
   */
  public setLeftTop( leftTop: Vector2 ): this {
    assert && assert( leftTop.isFinite(), 'leftTop should be a finite Vector2' );

    const currentLeftTop = this.getLeftTop();
    if ( currentLeftTop.isFinite() ) {
      this.translate( leftTop.minus( currentLeftTop ), true );
    }

    return this;
  }

  /**
   * See setLeftTop() for more information
   */
  public set leftTop( value: Vector2 ) {
    this.setLeftTop( value );
  }

  /**
   * See getLeftTop() for more information
   */
  public get leftTop(): Vector2 {
    return this.getLeftTop();
  }

  /**
   * Returns the upper-left corner of this node's bounds.
   */
  public getLeftTop(): Vector2 {
    return this.getBounds().getLeftTop();
  }

  /**
   * Sets the position of the center-top location of this node's bounds to the specified point.
   */
  public setCenterTop( centerTop: Vector2 ): this {
    assert && assert( centerTop.isFinite(), 'centerTop should be a finite Vector2' );

    const currentCenterTop = this.getCenterTop();
    if ( currentCenterTop.isFinite() ) {
      this.translate( centerTop.minus( currentCenterTop ), true );
    }

    return this;
  }

  /**
   * See setCenterTop() for more information
   */
  public set centerTop( value: Vector2 ) {
    this.setCenterTop( value );
  }

  /**
   * See getCenterTop() for more information
   */
  public get centerTop(): Vector2 {
    return this.getCenterTop();
  }

  /**
   * Returns the center-top location of this node's bounds.
   */
  public getCenterTop(): Vector2 {
    return this.getBounds().getCenterTop();
  }

  /**
   * Sets the position of the upper-right corner of this node's bounds to the specified point.
   */
  public setRightTop( rightTop: Vector2 ): this {
    assert && assert( rightTop.isFinite(), 'rightTop should be a finite Vector2' );

    const currentRightTop = this.getRightTop();
    if ( currentRightTop.isFinite() ) {
      this.translate( rightTop.minus( currentRightTop ), true );
    }

    return this;
  }

  /**
   * See setRightTop() for more information
   */
  public set rightTop( value: Vector2 ) {
    this.setRightTop( value );
  }

  /**
   * See getRightTop() for more information
   */
  public get rightTop(): Vector2 {
    return this.getRightTop();
  }

  /**
   * Returns the upper-right corner of this node's bounds.
   */
  public getRightTop(): Vector2 {
    return this.getBounds().getRightTop();
  }

  /**
   * Sets the position of the center-left of this node's bounds to the specified point.
   */
  public setLeftCenter( leftCenter: Vector2 ): this {
    assert && assert( leftCenter.isFinite(), 'leftCenter should be a finite Vector2' );

    const currentLeftCenter = this.getLeftCenter();
    if ( currentLeftCenter.isFinite() ) {
      this.translate( leftCenter.minus( currentLeftCenter ), true );
    }

    return this;
  }

  /**
   * See setLeftCenter() for more information
   */
  public set leftCenter( value: Vector2 ) {
    this.setLeftCenter( value );
  }

  /**
   * See getLeftCenter() for more information
   */
  public get leftCenter(): Vector2 {
    return this.getLeftCenter();
  }

  /**
   * Returns the center-left corner of this node's bounds.
   */
  public getLeftCenter(): Vector2 {
    return this.getBounds().getLeftCenter();
  }

  /**
   * Sets the center of this node's bounds to the specified point.
   */
  public setCenter( center: Vector2 ): this {
    assert && assert( center.isFinite(), 'center should be a finite Vector2' );

    const currentCenter = this.getCenter();
    if ( currentCenter.isFinite() ) {
      this.translate( center.minus( currentCenter ), true );
    }

    return this;
  }

  /**
   * See setCenter() for more information
   */
  public set center( value: Vector2 ) {
    this.setCenter( value );
  }

  /**
   * See getCenter() for more information
   */
  public get center(): Vector2 {
    return this.getCenter();
  }

  /**
   * Returns the center of this node's bounds.
   */
  public getCenter(): Vector2 {
    return this.getBounds().getCenter();
  }

  /**
   * Sets the position of the center-right of this node's bounds to the specified point.
   */
  public setRightCenter( rightCenter: Vector2 ): this {
    assert && assert( rightCenter.isFinite(), 'rightCenter should be a finite Vector2' );

    const currentRightCenter = this.getRightCenter();
    if ( currentRightCenter.isFinite() ) {
      this.translate( rightCenter.minus( currentRightCenter ), true );
    }

    return this;
  }

  /**
   * See setRightCenter() for more information
   */
  public set rightCenter( value: Vector2 ) {
    this.setRightCenter( value );
  }

  /**
   * See getRightCenter() for more information
   */
  public get rightCenter(): Vector2 {
    return this.getRightCenter();
  }

  /**
   * Returns the center-right of this node's bounds.
   */
  public getRightCenter(): Vector2 {
    return this.getBounds().getRightCenter();
  }

  /**
   * Sets the position of the lower-left corner of this node's bounds to the specified point.
   */
  public setLeftBottom( leftBottom: Vector2 ): this {
    assert && assert( leftBottom.isFinite(), 'leftBottom should be a finite Vector2' );

    const currentLeftBottom = this.getLeftBottom();
    if ( currentLeftBottom.isFinite() ) {
      this.translate( leftBottom.minus( currentLeftBottom ), true );
    }

    return this;
  }

  /**
   * See setLeftBottom() for more information
   */
  public set leftBottom( value: Vector2 ) {
    this.setLeftBottom( value );
  }

  /**
   * See getLeftBottom() for more information
   */
  public get leftBottom(): Vector2 {
    return this.getLeftBottom();
  }

  /**
   * Returns the lower-left corner of this node's bounds.
   */
  public getLeftBottom(): Vector2 {
    return this.getBounds().getLeftBottom();
  }

  /**
   * Sets the position of the center-bottom of this node's bounds to the specified point.
   */
  public setCenterBottom( centerBottom: Vector2 ): this {
    assert && assert( centerBottom.isFinite(), 'centerBottom should be a finite Vector2' );

    const currentCenterBottom = this.getCenterBottom();
    if ( currentCenterBottom.isFinite() ) {
      this.translate( centerBottom.minus( currentCenterBottom ), true );
    }

    return this;
  }

  /**
   * See setCenterBottom() for more information
   */
  public set centerBottom( value: Vector2 ) {
    this.setCenterBottom( value );
  }

  /**
   * See getCenterBottom() for more information
   */
  public get centerBottom(): Vector2 {
    return this.getCenterBottom();
  }

  /**
   * Returns the center-bottom of this node's bounds.
   */
  public getCenterBottom(): Vector2 {
    return this.getBounds().getCenterBottom();
  }

  /**
   * Sets the position of the lower-right corner of this node's bounds to the specified point.
   */
  public setRightBottom( rightBottom: Vector2 ): this {
    assert && assert( rightBottom.isFinite(), 'rightBottom should be a finite Vector2' );

    const currentRightBottom = this.getRightBottom();
    if ( currentRightBottom.isFinite() ) {
      this.translate( rightBottom.minus( currentRightBottom ), true );
    }

    return this;
  }

  /**
   * See setRightBottom() for more information
   */
  public set rightBottom( value: Vector2 ) {
    this.setRightBottom( value );
  }

  /**
   * See getRightBottom() for more information
   */
  public get rightBottom(): Vector2 {
    return this.getRightBottom();
  }

  /**
   * Returns the lower-right corner of this node's bounds.
   */
  public getRightBottom(): Vector2 {
    return this.getBounds().getRightBottom();
  }

  /**
   * Returns the width of this node's bounding box (in the parent coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getWidth(): number {
    return this.getBounds().getWidth();
  }

  /**
   * See getWidth() for more information
   */
  public get width(): number {
    return this.getWidth();
  }

  /**
   * Returns the height of this node's bounding box (in the parent coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getHeight(): number {
    return this.getBounds().getHeight();
  }

  /**
   * See getHeight() for more information
   */
  public get height(): number {
    return this.getHeight();
  }

  /**
   * Returns the width of this node's bounding box (in the local coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getLocalWidth(): number {
    return this.getLocalBounds().getWidth();
  }

  /**
   * See getLocalWidth() for more information
   */
  public get localWidth(): number {
    return this.getLocalWidth();
  }

  /**
   * Returns the height of this node's bounding box (in the local coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getLocalHeight(): number {
    return this.getLocalBounds().getHeight();
  }

  /**
   * See getLocalHeight() for more information
   */
  public get localHeight(): number {
    return this.getLocalHeight();
  }

  /**
   * Returns the X value of the left side of the bounding box of this Node (in the local coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getLocalLeft(): number {
    return this.getLocalBounds().minX;
  }

  /**
   * See getLeft() for more information
   */
  public get localLeft(): number {
    return this.getLocalLeft();
  }

  /**
   * Returns the X value of the right side of the bounding box of this Node (in the local coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getLocalRight(): number {
    return this.getLocalBounds().maxX;
  }

  /**
   * See getRight() for more information
   */
  public get localRight(): number {
    return this.getLocalRight();
  }

  /**
   * Returns the X value of this node's horizontal center (in the local coordinate frame)
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getLocalCenterX(): number {
    return this.getLocalBounds().getCenterX();
  }

  /**
   * See getCenterX() for more information
   */
  public get localCenterX(): number {
    return this.getLocalCenterX();
  }

  /**
   * Returns the Y value of this node's vertical center (in the local coordinate frame)
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getLocalCenterY(): number {
    return this.getLocalBounds().getCenterY();
  }

  /**
   * See getCenterX() for more information
   */
  public get localCenterY(): number {
    return this.getLocalCenterY();
  }

  /**
   * Returns the lowest Y value of this node's bounding box (in the local coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getLocalTop(): number {
    return this.getLocalBounds().minY;
  }

  /**
   * See getTop() for more information
   */
  public get localTop(): number {
    return this.getLocalTop();
  }

  /**
   * Returns the highest Y value of this node's bounding box (in the local coordinate frame).
   *
   * NOTE: This may require computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getLocalBottom(): number {
    return this.getLocalBounds().maxY;
  }

  /**
   * See getLocalBottom() for more information
   */
  public get localBottom(): number {
    return this.getLocalBottom();
  }

  /**
   * Returns the upper-left corner of this node's localBounds.
   */
  public getLocalLeftTop(): Vector2 {
    return this.getLocalBounds().getLeftTop();
  }

  /**
   * See getLocalLeftTop() for more information
   */
  public get localLeftTop(): Vector2 {
    return this.getLocalLeftTop();
  }

  /**
   * Returns the center-top location of this node's localBounds.
   */
  public getLocalCenterTop(): Vector2 {
    return this.getLocalBounds().getCenterTop();
  }

  /**
   * See getLocalCenterTop() for more information
   */
  public get localCenterTop(): Vector2 {
    return this.getLocalCenterTop();
  }

  /**
   * Returns the upper-right corner of this node's localBounds.
   */
  public getLocalRightTop(): Vector2 {
    return this.getLocalBounds().getRightTop();
  }

  /**
   * See getLocalRightTop() for more information
   */
  public get localRightTop(): Vector2 {
    return this.getLocalRightTop();
  }

  /**
   * Returns the center-left corner of this node's localBounds.
   */
  public getLocalLeftCenter(): Vector2 {
    return this.getLocalBounds().getLeftCenter();
  }

  /**
   * See getLocalLeftCenter() for more information
   */
  public get localLeftCenter(): Vector2 {
    return this.getLocalLeftCenter();
  }

  /**
   * Returns the center of this node's localBounds.
   */
  public getLocalCenter(): Vector2 {
    return this.getLocalBounds().getCenter();
  }

  /**
   * See getLocalCenter() for more information
   */
  public get localCenter(): Vector2 {
    return this.getLocalCenter();
  }

  /**
   * Returns the center-right of this node's localBounds.
   */
  public getLocalRightCenter(): Vector2 {
    return this.getLocalBounds().getRightCenter();
  }

  /**
   * See getLocalRightCenter() for more information
   */
  public get localRightCenter(): Vector2 {
    return this.getLocalRightCenter();
  }

  /**
   * Returns the lower-left corner of this node's localBounds.
   */
  public getLocalLeftBottom(): Vector2 {
    return this.getLocalBounds().getLeftBottom();
  }

  /**
   * See getLocalLeftBottom() for more information
   */
  public get localLeftBottom(): Vector2 {
    return this.getLocalLeftBottom();
  }

  /**
   * Returns the center-bottom of this node's localBounds.
   */
  public getLocalCenterBottom(): Vector2 {
    return this.getLocalBounds().getCenterBottom();
  }

  /**
   * See getLocalCenterBottom() for more information
   */
  public get localCenterBottom(): Vector2 {
    return this.getLocalCenterBottom();
  }

  /**
   * Returns the lower-right corner of this node's localBounds.
   */
  public getLocalRightBottom(): Vector2 {
    return this.getLocalBounds().getRightBottom();
  }

  /**
   * See getLocalRightBottom() for more information
   */
  public get localRightBottom(): Vector2 {
    return this.getLocalRightBottom();
  }

  /**
   * Returns the unique integral ID for this node.
   */
  public getId(): number {
    return this._id;
  }

  /**
   * See getId() for more information
   */
  public get id(): number {
    return this.getId();
  }

  /**
   * Called when our visibility Property changes values.
   */
  private onVisiblePropertyChange( visible: boolean ): void {

    // changing visibility can affect pickability pruning, which affects mouse/touch bounds
    this._picker.onVisibilityChange();

    if ( assertSlow ) { this._picker.audit(); }

    // Defined in ParallelDOM.js
    this._pdomDisplaysInfo.onVisibilityChange( visible );

    for ( let i = 0; i < this._parents.length; i++ ) {
      const parent = this._parents[ i ];
      if ( parent._excludeInvisibleChildrenFromBounds ) {
        parent.invalidateChildBounds();
      }
    }
  }

  /**
   * Sets what Property our visibleProperty is backed by, so that changes to this provided Property will change this
   * Node's visibility, and vice versa. This does not change this._visibleProperty. See TinyForwardingProperty.setTargetProperty()
   * for more info.
   *
   * NOTE For PhET-iO use:
   * All PhET-iO instrumented Nodes create their own instrumented visibleProperty (if one is not passed in as
   * an option). Once a Node's visibleProperty has been registered with PhET-iO, it cannot be "swapped out" for another.
   * If you need to "delay" setting an instrumented visibleProperty to this node, pass phetioVisiblePropertyInstrumented
   * to instrumentation call to this Node (where Tandem is provided).
   */
  public setVisibleProperty( newTarget: TReadOnlyProperty<boolean> | null ): this {
    return this._visibleProperty.setTargetProperty( this, VISIBLE_PROPERTY_TANDEM_NAME, newTarget as TProperty<boolean> );
  }

  /**
   * See setVisibleProperty() for more information
   */
  public set visibleProperty( property: TReadOnlyProperty<boolean> | null ) {
    this.setVisibleProperty( property );
  }

  /**
   * See getVisibleProperty() for more information
   */
  public get visibleProperty(): TProperty<boolean> {
    return this.getVisibleProperty();
  }


  /**
   * Get this Node's visibleProperty. Note! This is not the reciprocal of setVisibleProperty. Node.prototype._visibleProperty
   * is a TinyForwardingProperty, and is set up to listen to changes from the visibleProperty provided by
   * setVisibleProperty(), but the underlying reference does not change. This means the following:
   *     * const myNode = new Node();
   * const visibleProperty = new Property( false );
   * myNode.setVisibleProperty( visibleProperty )
   * => myNode.getVisibleProperty() !== visibleProperty (!!!!!!)
   *
   * Please use this with caution. See setVisibleProperty() for more information.
   */
  public getVisibleProperty(): TProperty<boolean> {
    return this._visibleProperty;
  }

  /**
   * Sets whether this Node is visible.  DO NOT override this as a way of adding additional behavior when a Node's
   * visibility changes, add a listener to this.visibleProperty instead.
   */
  public setVisible( visible: boolean ): this {
    this.visibleProperty.set( visible );
    return this;
  }

  /**
   * See setVisible() for more information
   */
  public set visible( value: boolean ) {
    this.setVisible( value );
  }

  /**
   * See isVisible() for more information
   */
  public get visible(): boolean {
    return this.isVisible();
  }

  /**
   * Returns whether this Node is visible.
   */
  public isVisible(): boolean {
    return this.visibleProperty.value;
  }

  /**
   * Use this to automatically create a forwarded, PhET-iO instrumented visibleProperty internal to Node.
   */
  public setPhetioVisiblePropertyInstrumented( phetioVisiblePropertyInstrumented: boolean ): this {
    return this._visibleProperty.setTargetPropertyInstrumented( phetioVisiblePropertyInstrumented, this );
  }

  /**
   * See setPhetioVisiblePropertyInstrumented() for more information
   */
  public set phetioVisiblePropertyInstrumented( value: boolean ) {
    this.setPhetioVisiblePropertyInstrumented( value );
  }

  /**
   * See getPhetioVisiblePropertyInstrumented() for more information
   */
  public get phetioVisiblePropertyInstrumented(): boolean {
    return this.getPhetioVisiblePropertyInstrumented();
  }

  public getPhetioVisiblePropertyInstrumented(): boolean {
    return this._visibleProperty.getTargetPropertyInstrumented();
  }

  /**
   * Swap the visibility of this node with another node. The Node that is made visible will receive keyboard focus
   * if it is focusable and the previously visible Node had focus.
   */
  public swapVisibility( otherNode: Node ): this {
    assert && assert( this.visible !== otherNode.visible );

    const visibleNode = this.visible ? this : otherNode;
    const invisibleNode = this.visible ? otherNode : this;

    // if the visible node has focus we will restore focus on the invisible Node once it is visible
    const visibleNodeFocused = visibleNode.focused;

    visibleNode.visible = false;
    invisibleNode.visible = true;

    if ( visibleNodeFocused && invisibleNode.focusable ) {
      invisibleNode.focus();
    }

    return this; // allow chaining
  }

  /**
   * Sets the opacity of this Node (and its sub-tree), where 0 is fully transparent, and 1 is fully opaque.  Values
   * outside of that range throw an Error.
   * @throws Error if opacity out of range
   */
  public setOpacity( opacity: number ): void {
    assert && assert( isFinite( opacity ), 'opacity should be a finite number' );

    if ( opacity < 0 || opacity > 1 ) {
      throw new Error( `opacity out of range: ${opacity}` );
    }

    this.opacityProperty.value = opacity;
  }

  /**
   * See setOpacity() for more information
   */
  public set opacity( value: number ) {
    this.setOpacity( value );
  }

  /**
   * See getOpacity() for more information
   */
  public get opacity(): number {
    return this.getOpacity();
  }

  /**
   * Returns the opacity of this node.
   */
  public getOpacity(): number {
    return this.opacityProperty.value;
  }

  /**
   * Sets the disabledOpacity of this Node (and its sub-tree), where 0 is fully transparent, and 1 is fully opaque.
   * Values outside of that range throw an Error.
   * @throws Error if disabledOpacity out of range
   */
  public setDisabledOpacity( disabledOpacity: number ): this {
    assert && assert( isFinite( disabledOpacity ), 'disabledOpacity should be a finite number' );

    if ( disabledOpacity < 0 || disabledOpacity > 1 ) {
      throw new Error( `disabledOpacity out of range: ${disabledOpacity}` );
    }

    this.disabledOpacityProperty.value = disabledOpacity;

    return this;
  }

  /**
   * See setDisabledOpacity() for more information
   */
  public set disabledOpacity( value: number ) {
    this.setDisabledOpacity( value );
  }

  /**
   * See getDisabledOpacity() for more information
   */
  public get disabledOpacity(): number {
    return this.getDisabledOpacity();
  }

  /**
   * Returns the disabledOpacity of this node.
   */
  public getDisabledOpacity(): number {
    return this.disabledOpacityProperty.value;
  }

  /**
   * Returns the opacity actually applied to the node.
   */
  public getEffectiveOpacity(): number {
    return this.opacityProperty.value * ( this.enabledProperty.value ? 1 : this.disabledOpacityProperty.value );
  }

  /**
   * See getDisabledOpacity() for more information
   */
  public get effectiveOpacity(): number {
    return this.getEffectiveOpacity();
  }

  /**
   * Called when our opacity or other filter changes values
   */
  private onOpacityPropertyChange(): void {
    this.filterChangeEmitter.emit();
  }

  /**
   * Called when our opacity or other filter changes values
   */
  private onDisabledOpacityPropertyChange(): void {
    if ( !this._enabledProperty.value ) {
      this.filterChangeEmitter.emit();
    }
  }

  /**
   * Sets the non-opacity filters for this Node.
   *
   * The default is an empty array (no filters). It should be an array of Filter objects, which will be effectively
   * applied in-order on this Node (and its subtree), and will be applied BEFORE opacity/clipping.
   *
   * NOTE: Some filters may decrease performance (and this may be platform-specific). Please read documentation for each
   * filter before using.
   *
   * Typical filter types to use are:
   * - Brightness
   * - Contrast
   * - DropShadow (EXPERIMENTAL)
   * - GaussianBlur (EXPERIMENTAL)
   * - Grayscale (Grayscale.FULL for the full effect)
   * - HueRotate
   * - Invert (Invert.FULL for the full effect)
   * - Saturate
   * - Sepia (Sepia.FULL for the full effect)
   *
   * Filter.js has more information in general on filters.
   */
  public setFilters( filters: Filter[] ): void {
    assert && assert( Array.isArray( filters ), 'filters should be an array' );
    assert && assert( _.every( filters, filter => filter instanceof Filter ), 'filters should consist of Filter objects only' );

    // We re-use the same array internally, so we don't reference a potentially-mutable array from outside.
    this._filters.length = 0;
    this._filters.push( ...filters );

    this.invalidateHint();
    this.filterChangeEmitter.emit();
  }

  /**
   * See setFilters() for more information
   */
  public set filters( value: Filter[] ) {
    this.setFilters( value );
  }

  /**
   * See getFilters() for more information
   */
  public get filters(): Filter[] {
    return this.getFilters();
  }

  /**
   * Returns the non-opacity filters for this Node.
   */
  public getFilters(): Filter[] {
    return this._filters.slice();
  }

  /**
   * Sets what Property our pickableProperty is backed by, so that changes to this provided Property will change this
   * Node's pickability, and vice versa. This does not change this._pickableProperty. See TinyForwardingProperty.setTargetProperty()
   * for more info.
   *
   * PhET-iO Instrumented Nodes do not by default create their own instrumented pickableProperty, even though Node.visibleProperty does.
   */
  public setPickableProperty( newTarget: TReadOnlyProperty<boolean | null> | null ): this {
    return this._pickableProperty.setTargetProperty( this, null, newTarget as TProperty<boolean | null> );
  }

  /**
   * See setPickableProperty() for more information
   */
  public set pickableProperty( property: TReadOnlyProperty<boolean | null> | null ) {
    this.setPickableProperty( property );
  }

  /**
   * See getPickableProperty() for more information
   */
  public get pickableProperty(): TProperty<boolean | null> {
    return this.getPickableProperty();
  }

  /**
   * Get this Node's pickableProperty. Note! This is not the reciprocal of setPickableProperty. Node.prototype._pickableProperty
   * is a TinyForwardingProperty, and is set up to listen to changes from the pickableProperty provided by
   * setPickableProperty(), but the underlying reference does not change. This means the following:
   * const myNode = new Node();
   * const pickableProperty = new Property( false );
   * myNode.setPickableProperty( pickableProperty )
   * => myNode.getPickableProperty() !== pickableProperty (!!!!!!)
   *
   * Please use this with caution. See setPickableProperty() for more information.
   */
  public getPickableProperty(): TProperty<boolean | null> {
    return this._pickableProperty;
  }

  /**
   * Sets whether this Node (and its subtree) will allow hit-testing (and thus user interaction), controlling what
   * Trail is returned from node.trailUnderPoint().
   *
   * Pickable can take one of three values:
   * - null: (default) pass-through behavior. Hit-testing will prune this subtree if there are no
   *         ancestors/descendants with either pickable: true set or with any input listeners.
   * - false: Hit-testing is pruned, nothing in this node or its subtree will respond to events or be picked.
   * - true: Hit-testing will not be pruned in this subtree, except for pickable: false cases.
   *
   * Hit testing is accomplished mainly with node.trailUnderPointer() and node.trailUnderPoint(), following the
   * above rules. Nodes that are not pickable (pruned) will not have input events targeted to them.
   *
   * The following rules (applied in the given order) determine whether a Node (really, a Trail) will receive input events:
   * 1. If the node or one of its ancestors has pickable: false OR is invisible, the Node *will not* receive events
   *    or hit testing.
   * 2. If the Node or one of its ancestors or descendants is pickable: true OR has an input listener attached, it
   *    *will* receive events or hit testing.
   * 3. Otherwise, it *will not* receive events or hit testing.
   *
   * This is useful for semi-transparent overlays or other visual elements that should be displayed but should not
   * prevent objects below from being manipulated by user input, and the default null value is used to increase
   * performance by ignoring areas that don't need user input.
   *
   * NOTE: If you want something to be picked "mouse is over it", but block input events even if there are listeners,
   *       then pickable:false is not appropriate, and inputEnabled:false is preferred.
   *
   * For a visual example of how pickability interacts with input listeners and visibility, see the notes at the
   * bottom of http://phetsims.github.io/scenery/doc/implementation-notes, or scenery/assets/pickability.svg.
   */
  public setPickable( pickable: boolean | null ): this {
    assert && assert( pickable === null || typeof pickable === 'boolean' );
    this._pickableProperty.set( pickable );

    return this;
  }

  /**
   * See setPickable() for more information
   */
  public set pickable( value: boolean | null ) {
    this.setPickable( value );
  }

  /**
   * See isPickable() for more information
   */
  public get pickable(): boolean | null {
    return this.isPickable();
  }

  /**
   * Returns the pickability of this node.
   */
  public isPickable(): boolean | null {
    return this._pickableProperty.value;
  }

  /**
   * Called when our pickableProperty changes values.
   */
  private onPickablePropertyChange( pickable: boolean | null, oldPickable: boolean | null ): void {
    this._picker.onPickableChange( oldPickable, pickable );
    if ( assertSlow ) { this._picker.audit(); }
    // TODO: invalidate the cursor somehow? #150
  }

  /**
   * Sets what Property our enabledProperty is backed by, so that changes to this provided Property will change this
   * Node's enabled, and vice versa. This does not change this._enabledProperty. See TinyForwardingProperty.setTargetProperty()
   * for more info.
   *
   *
   * NOTE For PhET-iO use:
   * All PhET-iO instrumented Nodes create their own instrumented enabledProperty (if one is not passed in as
   * an option). Once a Node's enabledProperty has been registered with PhET-iO, it cannot be "swapped out" for another.
   * If you need to "delay" setting an instrumented enabledProperty to this node, pass phetioEnabledPropertyInstrumented
   * to instrumentation call to this Node (where Tandem is provided).
   */
  public setEnabledProperty( newTarget: TReadOnlyProperty<boolean> | null ): this {
    return this._enabledProperty.setTargetProperty( this, ENABLED_PROPERTY_TANDEM_NAME, newTarget as TProperty<boolean> );
  }

  /**
   * See setEnabledProperty() for more information
   */
  public set enabledProperty( property: TReadOnlyProperty<boolean> | null ) {
    this.setEnabledProperty( property );
  }

  /**
   * See getEnabledProperty() for more information
   */
  public get enabledProperty(): TProperty<boolean> {
    return this.getEnabledProperty();
  }

  /**
   * Get this Node's enabledProperty. Note! This is not the reciprocal of setEnabledProperty. Node.prototype._enabledProperty
   * is a TinyForwardingProperty, and is set up to listen to changes from the enabledProperty provided by
   * setEnabledProperty(), but the underlying reference does not change. This means the following:
   * const myNode = new Node();
   * const enabledProperty = new Property( false );
   * myNode.setEnabledProperty( enabledProperty )
   * => myNode.getEnabledProperty() !== enabledProperty (!!!!!!)
   *
   * Please use this with caution. See setEnabledProperty() for more information.
   */
  public getEnabledProperty(): TProperty<boolean> {
    return this._enabledProperty;
  }

  /**
   * Use this to automatically create a forwarded, PhET-iO instrumented enabledProperty internal to Node. This is different
   * from visible because enabled by default doesn't not create this forwarded Property.
   */
  public setPhetioEnabledPropertyInstrumented( phetioEnabledPropertyInstrumented: boolean ): this {
    return this._enabledProperty.setTargetPropertyInstrumented( phetioEnabledPropertyInstrumented, this );
  }

  /**
   * See setPhetioEnabledPropertyInstrumented() for more information
   */
  public set phetioEnabledPropertyInstrumented( value: boolean ) {
    this.setPhetioEnabledPropertyInstrumented( value );
  }

  /**
   * See getPhetioEnabledPropertyInstrumented() for more information
   */
  public get phetioEnabledPropertyInstrumented(): boolean {
    return this.getPhetioEnabledPropertyInstrumented();
  }

  public getPhetioEnabledPropertyInstrumented(): boolean {
    return this._enabledProperty.getTargetPropertyInstrumented();
  }

  /**
   * Sets whether this Node is enabled
   */
  public setEnabled( enabled: boolean ): this {
    assert && assert( enabled === null || typeof enabled === 'boolean' );
    this._enabledProperty.set( enabled );

    return this;
  }

  /**
   * See setEnabled() for more information
   */
  public set enabled( value: boolean ) {
    this.setEnabled( value );
  }

  /**
   * See isEnabled() for more information
   */
  public get enabled(): boolean {
    return this.isEnabled();
  }

  /**
   * Returns the enabled of this node.
   */
  public isEnabled(): boolean {
    return this._enabledProperty.value;
  }

  /**
   * Called when enabledProperty changes values.
   * - override this to change the behavior of enabled
   */
  protected onEnabledPropertyChange( enabled: boolean ): void {
    !enabled && this.interruptSubtreeInput();
    this.inputEnabled = enabled;

    if ( this.disabledOpacityProperty.value !== 1 ) {
      this.filterChangeEmitter.emit();
    }
  }

  /**
   * Sets what Property our inputEnabledProperty is backed by, so that changes to this provided Property will change this whether this
   * Node's input is enabled, and vice versa. This does not change this._inputEnabledProperty. See TinyForwardingProperty.setTargetProperty()
   * for more info.
   *
   * NOTE For PhET-iO use:
   * All PhET-iO instrumented Nodes create their own instrumented inputEnabledProperty (if one is not passed in as
   * an option). Once a Node's inputEnabledProperty has been registered with PhET-iO, it cannot be "swapped out" for another.
   * If you need to "delay" setting an instrumented inputEnabledProperty to this node, pass phetioInputEnabledPropertyInstrumented
   * to instrumentation call to this Node (where Tandem is provided).
   */
  public setInputEnabledProperty( newTarget: TReadOnlyProperty<boolean> | null ): this {
    return this._inputEnabledProperty.setTargetProperty( this, INPUT_ENABLED_PROPERTY_TANDEM_NAME, newTarget as TProperty<boolean> );
  }

  /**
   * See setInputEnabledProperty() for more information
   */
  public set inputEnabledProperty( property: TReadOnlyProperty<boolean> | null ) {
    this.setInputEnabledProperty( property );
  }

  /**
   * See getInputEnabledProperty() for more information
   */
  public get inputEnabledProperty(): TProperty<boolean> {
    return this.getInputEnabledProperty();
  }

  /**
   * Get this Node's inputEnabledProperty. Note! This is not the reciprocal of setInputEnabledProperty. Node.prototype._inputEnabledProperty
   * is a TinyForwardingProperty, and is set up to listen to changes from the inputEnabledProperty provided by
   * setInputEnabledProperty(), but the underlying reference does not change. This means the following:
   * const myNode = new Node();
   * const inputEnabledProperty = new Property( false );
   * myNode.setInputEnabledProperty( inputEnabledProperty )
   * => myNode.getInputEnabledProperty() !== inputEnabledProperty (!!!!!!)
   *
   * Please use this with caution. See setInputEnabledProperty() for more information.
   */
  public getInputEnabledProperty(): TProperty<boolean> {
    return this._inputEnabledProperty;
  }

  /**
   * Use this to automatically create a forwarded, PhET-iO instrumented inputEnabledProperty internal to Node. This is different
   * from visible because inputEnabled by default doesn't not create this forwarded Property.
   */
  public setPhetioInputEnabledPropertyInstrumented( phetioInputEnabledPropertyInstrumented: boolean ): this {
    return this._inputEnabledProperty.setTargetPropertyInstrumented( phetioInputEnabledPropertyInstrumented, this );
  }

  /**
   * See setPhetioInputEnabledPropertyInstrumented() for more information
   */
  public set phetioInputEnabledPropertyInstrumented( value: boolean ) {
    this.setPhetioInputEnabledPropertyInstrumented( value );
  }

  /**
   * See getPhetioInputEnabledPropertyInstrumented() for more information
   */
  public get phetioInputEnabledPropertyInstrumented(): boolean {
    return this.getPhetioInputEnabledPropertyInstrumented();
  }

  public getPhetioInputEnabledPropertyInstrumented(): boolean {
    return this._inputEnabledProperty.getTargetPropertyInstrumented();
  }

  /**
   * Sets whether input is enabled for this Node and its subtree. If false, input event listeners will not be fired
   * on this Node or its descendants in the picked Trail. This does NOT effect picking (what Trail/nodes are under
   * a pointer), but only effects what listeners are fired.
   *
   * Additionally, this will affect cursor behavior. If inputEnabled=false, descendants of this Node will not be
   * checked when determining what cursor will be shown. Instead, if a pointer (e.g. mouse) is over a descendant,
   * this Node's cursor will be checked first, then ancestors will be checked as normal.
   */
  public setInputEnabled( inputEnabled: boolean ): void {
    this.inputEnabledProperty.value = inputEnabled;
  }

  /**
   * See setInputEnabled() for more information
   */
  public set inputEnabled( value: boolean ) {
    this.setInputEnabled( value );
  }

  /**
   * See isInputEnabled() for more information
   */
  public get inputEnabled(): boolean {
    return this.isInputEnabled();
  }

  /**
   * Returns whether input is enabled for this Node and its subtree. See setInputEnabled for more documentation.
   */
  public isInputEnabled(): boolean {
    return this.inputEnabledProperty.value;
  }

  /**
   * Sets all of the input listeners attached to this Node.
   *
   * This is equivalent to removing all current input listeners with removeInputListener() and adding all new
   * listeners (in order) with addInputListener().
   */
  public setInputListeners( inputListeners: TInputListener[] ): this {
    assert && assert( Array.isArray( inputListeners ) );

    // Remove all old input listeners
    while ( this._inputListeners.length ) {
      this.removeInputListener( this._inputListeners[ 0 ] );
    }

    // Add in all new input listeners
    for ( let i = 0; i < inputListeners.length; i++ ) {
      this.addInputListener( inputListeners[ i ] );
    }

    return this;
  }

  /**
   * See setInputListeners() for more information
   */
  public set inputListeners( value: TInputListener[] ) {
    this.setInputListeners( value );
  }

  /**
   * See getInputListeners() for more information
   */
  public get inputListeners(): TInputListener[] {
    return this.getInputListeners();
  }

  /**
   * Returns a copy of all of our input listeners.
   */
  public getInputListeners(): TInputListener[] {
    return this._inputListeners.slice( 0 ); // defensive copy
  }

  /**
   * Sets the CSS cursor string that should be used when the mouse is over this node. null is the default, and
   * indicates that ancestor nodes (or the browser default) should be used.
   *
   * @param cursor - A CSS cursor string, like 'pointer', or 'none' - Examples are:
   * auto default none inherit help pointer progress wait crosshair text vertical-text alias copy move no-drop not-allowed
   * e-resize n-resize w-resize s-resize nw-resize ne-resize se-resize sw-resize ew-resize ns-resize nesw-resize nwse-resize
   * context-menu cell col-resize row-resize all-scroll url( ... ) --> does it support data URLs?
   */
  public setCursor( cursor: string | null ): void {

    // TODO: consider a mapping of types to set reasonable defaults

    // allow the 'auto' cursor type to let the ancestors or scene pick the cursor type
    this._cursor = cursor === 'auto' ? null : cursor;
  }

  /**
   * See setCursor() for more information
   */
  public set cursor( value: string | null ) {
    this.setCursor( value );
  }

  /**
   * See getCursor() for more information
   */
  public get cursor(): string | null {
    return this.getCursor();
  }

  /**
   * Returns the CSS cursor string for this node, or null if there is no cursor specified.
   */
  public getCursor(): string | null {
    return this._cursor;
  }

  /**
   * Returns the CSS cursor that could be applied either by this Node itself, or from any of its input listeners'
   * preferences. (scenery-internal)
   */
  public getEffectiveCursor(): string | null {
    if ( this._cursor ) {
      return this._cursor;
    }

    for ( let i = 0; i < this._inputListeners.length; i++ ) {
      const inputListener = this._inputListeners[ i ];

      if ( inputListener.cursor ) {
        return inputListener.cursor;
      }
    }

    return null;
  }

  /**
   * Sets the hit-tested mouse area for this Node (see constructor for more advanced documentation). Use null for the
   * default behavior.
   */
  public setMouseArea( area: Shape | Bounds2 | null ): this {
    assert && assert( area === null || area instanceof Shape || area instanceof Bounds2, 'mouseArea needs to be a phet.kite.Shape, phet.dot.Bounds2, or null' );

    if ( this._mouseArea !== area ) {
      this._mouseArea = area; // TODO: could change what is under the mouse, invalidate!

      this._picker.onMouseAreaChange();
      if ( assertSlow ) { this._picker.audit(); }
    }

    return this;
  }

  /**
   * See setMouseArea() for more information
   */
  public set mouseArea( value: Shape | Bounds2 | null ) {
    this.setMouseArea( value );
  }

  /**
   * See getMouseArea() for more information
   */
  public get mouseArea(): Shape | Bounds2 | null {
    return this.getMouseArea();
  }

  /**
   * Returns the hit-tested mouse area for this node.
   */
  public getMouseArea(): Shape | Bounds2 | null {
    return this._mouseArea;
  }

  /**
   * Sets the hit-tested touch area for this Node (see constructor for more advanced documentation). Use null for the
   * default behavior.
   */
  public setTouchArea( area: Shape | Bounds2 | null ): this {
    assert && assert( area === null || area instanceof Shape || area instanceof Bounds2, 'touchArea needs to be a phet.kite.Shape, phet.dot.Bounds2, or null' );

    if ( this._touchArea !== area ) {
      this._touchArea = area; // TODO: could change what is under the touch, invalidate!

      this._picker.onTouchAreaChange();
      if ( assertSlow ) { this._picker.audit(); }
    }

    return this;
  }

  /**
   * See setTouchArea() for more information
   */
  public set touchArea( value: Shape | Bounds2 | null ) {
    this.setTouchArea( value );
  }

  /**
   * See getTouchArea() for more information
   */
  public get touchArea(): Shape | Bounds2 | null {
    return this.getTouchArea();
  }

  /**
   * Returns the hit-tested touch area for this node.
   */
  public getTouchArea(): Shape | Bounds2 | null {
    return this._touchArea;
  }

  /**
   * Sets a clipped shape where only content in our local coordinate frame that is inside the clip area will be shown
   * (anything outside is fully transparent).
   */
  public setClipArea( shape: Shape | null ): void {
    assert && assert( shape === null || shape instanceof Shape, 'clipArea needs to be a phet.kite.Shape, or null' );

    if ( this.clipArea !== shape ) {
      this.clipAreaProperty.value = shape;

      this.invalidateBounds();
      this._picker.onClipAreaChange();

      if ( assertSlow ) { this._picker.audit(); }
    }
  }

  /**
   * See setClipArea() for more information
   */
  public set clipArea( value: Shape | null ) {
    this.setClipArea( value );
  }

  /**
   * See getClipArea() for more information
   */
  public get clipArea(): Shape | null {
    return this.getClipArea();
  }

  /**
   * Returns the clipped area for this node.
   */
  public getClipArea(): Shape | null {
    return this.clipAreaProperty.value;
  }

  /**
   * Returns whether this Node has a clip area.
   */
  public hasClipArea(): boolean {
    return this.clipArea !== null;
  }

  /**
   * Sets what self renderers (and other bitmask flags) are supported by this node.
   */
  protected setRendererBitmask( bitmask: number ): void {
    assert && assert( isFinite( bitmask ) );

    if ( bitmask !== this._rendererBitmask ) {
      this._rendererBitmask = bitmask;

      this._rendererSummary.selfChange();

      this.instanceRefreshEmitter.emit();
    }
  }

  /**
   * Meant to be overridden, so that it can be called to ensure that the renderer bitmask will be up-to-date.
   */
  public invalidateSupportedRenderers(): void {
    // see docs
  }

  /*---------------------------------------------------------------------------*
   * Hints
   *----------------------------------------------------------------------------*/

  /**
   * When ANY hint changes, we refresh everything currently (for safety, this may be possible to make more specific
   * in the future, but hint changes are not particularly common performance bottleneck).
   */
  private invalidateHint(): void {
    this.rendererSummaryRefreshEmitter.emit();
    this.instanceRefreshEmitter.emit();
  }

  /**
   * Sets a preferred renderer for this Node and its sub-tree. Scenery will attempt to use this renderer under here
   * unless it isn't supported, OR another preferred renderer is set as a closer ancestor. Acceptable values are:
   * - null (default, no preference)
   * - 'canvas'
   * - 'svg'
   * - 'dom'
   * - 'webgl'
   */
  public setRenderer( renderer: RendererType ): void {
    assert && assert( renderer === null || renderer === 'canvas' || renderer === 'svg' || renderer === 'dom' || renderer === 'webgl',
      'Renderer input should be null, or one of: "canvas", "svg", "dom" or "webgl".' );

    let newRenderer = 0;
    if ( renderer === 'canvas' ) {
      newRenderer = Renderer.bitmaskCanvas;
    }
    else if ( renderer === 'svg' ) {
      newRenderer = Renderer.bitmaskSVG;
    }
    else if ( renderer === 'dom' ) {
      newRenderer = Renderer.bitmaskDOM;
    }
    else if ( renderer === 'webgl' ) {
      newRenderer = Renderer.bitmaskWebGL;
    }
    assert && assert( ( renderer === null ) === ( newRenderer === 0 ),
      'We should only end up with no actual renderer if renderer is null' );

    if ( this._hints.renderer !== newRenderer ) {
      this._hints.renderer = newRenderer;

      this.invalidateHint();
    }
  }

  /**
   * See setRenderer() for more information
   */
  public set renderer( value: RendererType ) {
    this.setRenderer( value );
  }

  /**
   * See getRenderer() for more information
   */
  public get renderer(): RendererType {
    return this.getRenderer();
  }

  /**
   * Returns the preferred renderer (if any) of this node, as a string.
   */
  public getRenderer(): RendererType {
    if ( this._hints.renderer === 0 ) {
      return null;
    }
    else if ( this._hints.renderer === Renderer.bitmaskCanvas ) {
      return 'canvas';
    }
    else if ( this._hints.renderer === Renderer.bitmaskSVG ) {
      return 'svg';
    }
    else if ( this._hints.renderer === Renderer.bitmaskDOM ) {
      return 'dom';
    }
    else if ( this._hints.renderer === Renderer.bitmaskWebGL ) {
      return 'webgl';
    }
    assert && assert( false, 'Seems to be an invalid renderer?' );
    return null;
  }

  /**
   * Sets whether or not Scenery will try to put this Node (and its descendants) into a separate SVG/Canvas/WebGL/etc.
   * layer, different from other siblings or other nodes. Can be used for performance purposes.
   */
  public setLayerSplit( split: boolean ): void {
    if ( split !== this._hints.layerSplit ) {
      this._hints.layerSplit = split;

      this.invalidateHint();
    }
  }

  /**
   * See setLayerSplit() for more information
   */
  public set layerSplit( value: boolean ) {
    this.setLayerSplit( value );
  }

  /**
   * See isLayerSplit() for more information
   */
  public get layerSplit(): boolean {
    return this.isLayerSplit();
  }

  /**
   * Returns whether the layerSplit performance flag is set.
   */
  public isLayerSplit(): boolean {
    return this._hints.layerSplit;
  }

  /**
   * Sets whether or not Scenery will take into account that this Node plans to use opacity. Can have performance
   * gains if there need to be multiple layers for this node's descendants.
   */
  public setUsesOpacity( usesOpacity: boolean ): void {
    if ( usesOpacity !== this._hints.usesOpacity ) {
      this._hints.usesOpacity = usesOpacity;

      this.invalidateHint();
    }
  }

  /**
   * See setUsesOpacity() for more information
   */
  public set usesOpacity( value: boolean ) {
    this.setUsesOpacity( value );
  }

  /**
   * See getUsesOpacity() for more information
   */
  public get usesOpacity(): boolean {
    return this.getUsesOpacity();
  }

  /**
   * Returns whether the usesOpacity performance flag is set.
   */
  public getUsesOpacity(): boolean {
    return this._hints.usesOpacity;
  }

  /**
   * Sets a flag for whether whether the contents of this Node and its children should be displayed in a separate
   * DOM element that is transformed with CSS transforms. It can have potential speedups, since the browser may not
   * have to re-rasterize contents when it is animated.
   */
  public setCSSTransform( cssTransform: boolean ): void {
    if ( cssTransform !== this._hints.cssTransform ) {
      this._hints.cssTransform = cssTransform;

      this.invalidateHint();
    }
  }

  /**
   * See setCSSTransform() for more information
   */
  public set cssTransform( value: boolean ) {
    this.setCSSTransform( value );
  }

  /**
   * See isCSSTransformed() for more information
   */
  public get cssTransform(): boolean {
    return this.isCSSTransformed();
  }

  /**
   * Returns whether the cssTransform performance flag is set.
   */
  public isCSSTransformed(): boolean {
    return this._hints.cssTransform;
  }

  /**
   * Sets a performance flag for whether layers/DOM elements should be excluded (or included) when things are
   * invisible. The default is false, and invisible content is in the DOM, but hidden.
   */
  public setExcludeInvisible( excludeInvisible: boolean ): void {
    if ( excludeInvisible !== this._hints.excludeInvisible ) {
      this._hints.excludeInvisible = excludeInvisible;

      this.invalidateHint();
    }
  }

  /**
   * See setExcludeInvisible() for more information
   */
  public set excludeInvisible( value: boolean ) {
    this.setExcludeInvisible( value );
  }

  /**
   * See isExcludeInvisible() for more information
   */
  public get excludeInvisible(): boolean {
    return this.isExcludeInvisible();
  }

  /**
   * Returns whether the excludeInvisible performance flag is set.
   */
  public isExcludeInvisible(): boolean {
    return this._hints.excludeInvisible;
  }

  /**
   * If this is set to true, child nodes that are invisible will NOT contribute to the bounds of this node.
   *
   * The default is for child nodes bounds' to be included in this node's bounds, but that would in general be a
   * problem for layout containers or other situations, see https://github.com/phetsims/joist/issues/608.
   */
  public setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds: boolean ): void {
    if ( excludeInvisibleChildrenFromBounds !== this._excludeInvisibleChildrenFromBounds ) {
      this._excludeInvisibleChildrenFromBounds = excludeInvisibleChildrenFromBounds;

      this.invalidateBounds();
    }
  }

  /**
   * See setExcludeInvisibleChildrenFromBounds() for more information
   */
  public set excludeInvisibleChildrenFromBounds( value: boolean ) {
    this.setExcludeInvisibleChildrenFromBounds( value );
  }

  /**
   * See isExcludeInvisibleChildrenFromBounds() for more information
   */
  public get excludeInvisibleChildrenFromBounds(): boolean {
    return this.isExcludeInvisibleChildrenFromBounds();
  }

  /**
   * Returns whether the excludeInvisibleChildrenFromBounds flag is set, see
   * setExcludeInvisibleChildrenFromBounds() for documentation.
   */
  public isExcludeInvisibleChildrenFromBounds(): boolean {
    return this._excludeInvisibleChildrenFromBounds;
  }

  /**
   * Sets options that are provided to layout managers in order to customize positioning of this node.
   */
  public setLayoutOptions( layoutOptions: TLayoutOptions | null ): void {
    assert && assert( layoutOptions === null || ( typeof layoutOptions === 'object' && Object.getPrototypeOf( layoutOptions ) === Object.prototype ),
      'layoutOptions should be null or an plain options-style object' );

    if ( layoutOptions !== this._layoutOptions ) {
      this._layoutOptions = layoutOptions;

      this.layoutOptionsChangedEmitter.emit();
    }
  }

  public set layoutOptions( value: TLayoutOptions | null ) {
    this.setLayoutOptions( value );
  }

  public get layoutOptions(): TLayoutOptions | null {
    return this.getLayoutOptions();
  }

  public getLayoutOptions(): TLayoutOptions | null {
    return this._layoutOptions;
  }

  public mutateLayoutOptions( layoutOptions?: TLayoutOptions ): void {
    this.layoutOptions = optionize3<TLayoutOptions, EmptySelfOptions, TLayoutOptions>()( {}, this.layoutOptions || {}, layoutOptions );
  }

  // Defaults indicating that we don't mix in WidthSizable/HeightSizable
  public get widthSizable(): boolean { return false; }

  public get heightSizable(): boolean { return false; }

  public get extendsWidthSizable(): boolean { return false; }

  public get extendsHeightSizable(): boolean { return false; }

  public get extendsSizable(): boolean { return false; }

  /**
   * Sets the preventFit performance flag.
   */
  public setPreventFit( preventFit: boolean ): void {
    if ( preventFit !== this._hints.preventFit ) {
      this._hints.preventFit = preventFit;

      this.invalidateHint();
    }
  }

  /**
   * See setPreventFit() for more information
   */
  public set preventFit( value: boolean ) {
    this.setPreventFit( value );
  }

  /**
   * See isPreventFit() for more information
   */
  public get preventFit(): boolean {
    return this.isPreventFit();
  }

  /**
   * Returns whether the preventFit performance flag is set.
   */
  public isPreventFit(): boolean {
    return this._hints.preventFit;
  }

  /**
   * Sets whether there is a custom WebGL scale applied to the Canvas, and if so what scale.
   */
  public setWebGLScale( webglScale: number | null ): void {
    assert && assert( webglScale === null || ( typeof webglScale === 'number' && isFinite( webglScale ) ) );

    if ( webglScale !== this._hints.webglScale ) {
      this._hints.webglScale = webglScale;

      this.invalidateHint();
    }
  }

  /**
   * See setWebGLScale() for more information
   */
  public set webglScale( value: number | null ) {
    this.setWebGLScale( value );
  }

  /**
   * See getWebGLScale() for more information
   */
  public get webglScale(): number | null {
    return this.getWebGLScale();
  }

  /**
   * Returns the value of the webglScale performance flag.
   */
  public getWebGLScale(): number | null {
    return this._hints.webglScale;
  }

  /*---------------------------------------------------------------------------*
   * Trail operations
   *----------------------------------------------------------------------------*/

  /**
   * Returns the one Trail that starts from a node with no parents (or if the predicate is present, a Node that
   * satisfies it), and ends at this node. If more than one Trail would satisfy these conditions, an assertion is
   * thrown (please use getTrails() for those cases).
   *
   * @param [predicate] - If supplied, we will only return trails rooted at a Node that satisfies predicate( node ) == true
   */
  public getUniqueTrail( predicate?: ( node: Node ) => boolean ): Trail {

    // Without a predicate, we'll be able to bail out the instant we hit a Node with 2+ parents, and it makes the
    // logic easier.
    if ( !predicate ) {
      const trail = new Trail();

      // eslint-disable-next-line @typescript-eslint/no-this-alias
      let node: Node = this; // eslint-disable-line consistent-this

      while ( node ) {
        assert && assert( node._parents.length <= 1,
          `getUniqueTrail found a Node with ${node._parents.length} parents.` );

        trail.addAncestor( node );
        node = node._parents[ 0 ]; // should be undefined if there aren't any parents
      }

      return trail;
    }
    // With a predicate, we need to explore multiple parents (since the predicate may filter out all but one)
    else {
      const trails = this.getTrails( predicate );

      assert && assert( trails.length === 1,
        `getUniqueTrail found ${trails.length} matching trails for the predicate` );

      return trails[ 0 ];
    }
  }

  /**
   * Returns a Trail rooted at rootNode and ends at this node. Throws an assertion if the number of trails that match
   * this condition isn't exactly 1.
   */
  public getUniqueTrailTo( rootNode: Node ): Trail {
    return this.getUniqueTrail( node => rootNode === node );
  }

  /**
   * Returns an array of all Trails that start from nodes with no parent (or if a predicate is present, those that
   * satisfy the predicate), and ends at this node.
   *
   * @param [predicate] - If supplied, we will only return Trails rooted at nodes that satisfy predicate( node ) == true.
   */
  public getTrails( predicate?: ( node: Node ) => boolean ): Trail[] {
    predicate = predicate || Node.defaultTrailPredicate;

    const trails: Trail[] = [];
    const trail = new Trail( this );
    Trail.appendAncestorTrailsWithPredicate( trails, trail, predicate );

    return trails;
  }

  /**
   * Returns an array of all Trails rooted at rootNode and end at this node.
   */
  public getTrailsTo( rootNode: Node ): Trail[] {
    return this.getTrails( node => node === rootNode );
  }

  /**
   * Returns an array of all Trails rooted at this Node and end with nodes with no children (or if a predicate is
   * present, those that satisfy the predicate).
   *
   * @param [predicate] - If supplied, we will only return Trails ending at nodes that satisfy predicate( node ) == true.
   */
  public getLeafTrails( predicate?: ( node: Node ) => boolean ): Trail[] {
    predicate = predicate || Node.defaultLeafTrailPredicate;

    const trails: Trail[] = [];
    const trail = new Trail( this );
    Trail.appendDescendantTrailsWithPredicate( trails, trail, predicate );

    return trails;
  }

  /**
   * Returns an array of all Trails rooted at this Node and end with leafNode.
   */
  public getLeafTrailsTo( leafNode: Node ): Trail[] {
    return this.getLeafTrails( node => node === leafNode );
  }

  /**
   * Returns a Trail rooted at this node and ending at a Node that has no children (or if a predicate is provided, a
   * Node that satisfies the predicate). If more than one trail matches this description, an assertion will be fired.
   *
   * @param [predicate] - If supplied, we will return a Trail that ends with a Node that satisfies predicate( node ) == true
   */
  public getUniqueLeafTrail( predicate?: ( node: Node ) => boolean ): Trail {
    const trails = this.getLeafTrails( predicate );

    assert && assert( trails.length === 1,
      `getUniqueLeafTrail found ${trails.length} matching trails for the predicate` );

    return trails[ 0 ];
  }

  /**
   * Returns a Trail rooted at this Node and ending at leafNode. If more than one trail matches this description,
   * an assertion will be fired.
   */
  public getUniqueLeafTrailTo( leafNode: Node ): Trail {
    return this.getUniqueLeafTrail( node => node === leafNode );
  }

  /**
   * Returns all nodes in the connected component, returned in an arbitrary order, including nodes that are ancestors
   * of this node.
   */
  public getConnectedNodes(): Node[] {
    const result: Node[] = [];
    let fresh = this._children.concat( this._parents ).concat( this );
    while ( fresh.length ) {
      const node = fresh.pop()!;
      if ( !_.includes( result, node ) ) {
        result.push( node );
        fresh = fresh.concat( node._children, node._parents );
      }
    }
    return result;
  }

  /**
   * Returns all nodes in the subtree with this Node as its root, returned in an arbitrary order. Like
   * getConnectedNodes, but doesn't include parents.
   */
  public getSubtreeNodes(): Node[] {
    const result: Node[] = [];
    let fresh = this._children.concat( this );
    while ( fresh.length ) {
      const node = fresh.pop()!;
      if ( !_.includes( result, node ) ) {
        result.push( node );
        fresh = fresh.concat( node._children );
      }
    }
    return result;
  }

  /**
   * Returns all nodes that are connected to this node, sorted in topological order.
   */
  public getTopologicallySortedNodes(): Node[] {
    // see http://en.wikipedia.org/wiki/Topological_sorting
    const edges: Record<string, Record<string, boolean>> = {};
    const s: Node[] = [];
    const l: Node[] = [];
    let n: Node;
    _.each( this.getConnectedNodes(), node => {
      edges[ node.id ] = {};
      _.each( node._children, m => {
        edges[ node.id ][ m.id ] = true;
      } );
      if ( !node.parents.length ) {
        s.push( node );
      }
    } );

    function handleChild( m: Node ): void {
      delete edges[ n.id ][ m.id ];
      if ( _.every( edges, children => !children[ m.id ] ) ) {
        // there are no more edges to m
        s.push( m );
      }
    }

    while ( s.length ) {
      n = s.pop()!;
      l.push( n );

      _.each( n._children, handleChild );
    }

    // ensure that there are no edges left, since then it would contain a circular reference
    assert && assert( _.every( edges, children => _.every( children, final => false ) ), 'circular reference check' );

    return l;
  }

  /**
   * Returns whether this.addChild( child ) will not cause circular references.
   */
  public canAddChild( child: Node ): boolean {
    if ( this === child || _.includes( this._children, child ) ) {
      return false;
    }

    // see http://en.wikipedia.org/wiki/Topological_sorting
    // TODO: remove duplication with above handling?
    const edges: Record<string, Record<string, boolean>> = {};
    const s: Node[] = [];
    const l: Node[] = [];
    let n: Node;
    _.each( this.getConnectedNodes().concat( child.getConnectedNodes() ), node => {
      edges[ node.id ] = {};
      _.each( node._children, m => {
        edges[ node.id ][ m.id ] = true;
      } );
      if ( !node.parents.length && node !== child ) {
        s.push( node );
      }
    } );
    edges[ this.id ][ child.id ] = true; // add in our 'new' edge
    function handleChild( m: Node ): void {
      delete edges[ n.id ][ m.id ];
      if ( _.every( edges, children => !children[ m.id ] ) ) {
        // there are no more edges to m
        s.push( m );
      }
    }

    while ( s.length ) {
      n = s.pop()!;
      l.push( n );

      _.each( n._children, handleChild );

      // handle our new edge
      if ( n === this ) {
        handleChild( child );
      }
    }

    // ensure that there are no edges left, since then it would contain a circular reference
    return _.every( edges, children => _.every( children, final => false ) );
  }

  /**
   * To be overridden in paintable Node types. Should hook into the drawable's prototype (presumably).
   *
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   *
   * @param wrapper
   * @param matrix - The transformation matrix already applied to the context.
   */
  protected canvasPaintSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ): void {
    // See subclass for implementation
  }

  /**
   * Renders this Node only (its self) into the Canvas wrapper, in its local coordinate frame.
   *
   * @param wrapper
   * @param matrix - The transformation matrix already applied to the context.
   */
  public renderToCanvasSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ): void {
    if ( this.isPainted() && ( this._rendererBitmask & Renderer.bitmaskCanvas ) ) {
      this.canvasPaintSelf( wrapper, matrix );
    }
  }

  /**
   * Renders this Node and its descendants into the Canvas wrapper.
   *
   * @param wrapper
   * @param [matrix] - Optional transform to be applied
   */
  public renderToCanvasSubtree( wrapper: CanvasContextWrapper, matrix?: Matrix3 ): void {
    matrix = matrix || Matrix3.identity();

    wrapper.resetStyles();

    this.renderToCanvasSelf( wrapper, matrix );
    for ( let i = 0; i < this._children.length; i++ ) {
      const child = this._children[ i ];

      // Ignore invalid (empty) bounds, since this would show nothing (and we couldn't compute fitted bounds for it).
      if ( child.isVisible() && child.bounds.isValid() ) {

        // For anything filter-like, we'll need to create a Canvas, render our child's content into that Canvas,
        // and then (applying the filter) render that into the Canvas provided.
        const requiresScratchCanvas = child.effectiveOpacity !== 1 || child.clipArea || child._filters.length;

        wrapper.context.save();
        matrix.multiplyMatrix( child._transform.getMatrix() );
        matrix.canvasSetTransform( wrapper.context );
        if ( requiresScratchCanvas ) {
          // We'll attempt to fit the Canvas to the content to minimize memory use, see
          // https://github.com/phetsims/function-builder/issues/148

          // We're going to ignore content outside our wrapper context's canvas.
          // Added padding and round-out for cases where Canvas bounds might not be fully accurate
          // The matrix already includes the child's transform (so we use localBounds).
          // We won't go outside our parent canvas' bounds, since this would be a waste of memory (wouldn't be written)
          // The round-out will make sure we have pixel alignment, so that we won't get blurs or aliasing/blitting
          // effects when copying things over.
          const childCanvasBounds = child.localBounds.transformed( matrix ).dilate( 4 ).roundOut().constrainBounds(
            scratchBounds2Extra.setMinMax( 0, 0, wrapper.canvas.width, wrapper.canvas.height )
          );

          if ( childCanvasBounds.width > 0 && childCanvasBounds.height > 0 ) {
            const canvas = document.createElement( 'canvas' );

            // We'll set our Canvas to the fitted width, and will handle the offsets below.
            canvas.width = childCanvasBounds.width;
            canvas.height = childCanvasBounds.height;
            const context = canvas.getContext( '2d' )!;
            const childWrapper = new CanvasContextWrapper( canvas, context );

            // After our ancestor transform is applied, we'll need to apply another offset for fitted Canvas. We'll
            // need to pass this to descendants AND apply it to the sub-context.
            const subMatrix = matrix.copy().prependTranslation( -childCanvasBounds.minX, -childCanvasBounds.minY );

            subMatrix.canvasSetTransform( context );
            child.renderToCanvasSubtree( childWrapper, subMatrix );

            wrapper.context.save();
            if ( child.clipArea ) {
              wrapper.context.beginPath();
              child.clipArea.writeToContext( wrapper.context );
              wrapper.context.clip();
            }
            wrapper.context.setTransform( 1, 0, 0, 1, 0, 0 ); // identity
            wrapper.context.globalAlpha = child.effectiveOpacity;

            let setFilter = false;
            if ( child._filters.length ) {
              // Filters shouldn't be too often, so less concerned about the GC here (and this is so much easier to read).
              // Performance bottleneck for not using this fallback style, so we're allowing it for Chrome even though
              // the visual differences may be present, see https://github.com/phetsims/scenery/issues/1139
              if ( Features.canvasFilter && _.every( child._filters, filter => filter.isDOMCompatible() ) ) {
                wrapper.context.filter = child._filters.map( filter => filter.getCSSFilterString() ).join( ' ' );
                setFilter = true;
              }
              else {
                child._filters.forEach( filter => filter.applyCanvasFilter( childWrapper ) );
              }
            }

            // The inverse transform is applied to handle fitting
            wrapper.context.drawImage( canvas, childCanvasBounds.minX, childCanvasBounds.minY );
            wrapper.context.restore();
            if ( setFilter ) {
              wrapper.context.filter = 'none';
            }
          }
        }
        else {
          child.renderToCanvasSubtree( wrapper, matrix );
        }
        matrix.multiplyMatrix( child._transform.getInverse() );
        wrapper.context.restore();
      }
    }
  }

  /**
   * @deprecated
   * Render this Node to the Canvas (clearing it first)
   */
  public renderToCanvas( canvas: HTMLCanvasElement, context: CanvasRenderingContext2D, callback?: () => void, backgroundColor?: string ): void {

    assert && deprecationWarning( 'Node.renderToCanvas() is deprecated, please use Node.rasterized() instead' );

    // should basically reset everything (and clear the Canvas)
    canvas.width = canvas.width; // eslint-disable-line no-self-assign

    if ( backgroundColor ) {
      context.fillStyle = backgroundColor;
      context.fillRect( 0, 0, canvas.width, canvas.height );
    }

    const wrapper = new CanvasContextWrapper( canvas, context );

    this.renderToCanvasSubtree( wrapper, Matrix3.identity() );

    callback && callback(); // this was originally asynchronous, so we had a callback
  }

  /**
   * Renders this Node to an HTMLCanvasElement. If toCanvas( callback ) is used, the canvas will contain the node's
   * entire bounds (if no x/y/width/height is provided)
   *
   * @param callback - callback( canvas, x, y, width, height ) is called, where x,y are computed if not specified.
   * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param [width] - The width of the Canvas output
   * @param [height] - The height of the Canvas output
   */
  public toCanvas( callback: ( canvas: HTMLCanvasElement, x: number, y: number, width: number, height: number ) => void, x?: number, y?: number, width?: number, height?: number ): void {
    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    const padding = 2; // padding used if x and y are not set

    // for now, we add an unpleasant hack around Text and safe bounds in general. We don't want to add another Bounds2 object per Node for now.
    const bounds = this.getBounds().union( this.localToParentBounds( this.getSafeSelfBounds() ) );
    assert && assert( !bounds.isEmpty() ||
                      ( x !== undefined && y !== undefined && width !== undefined && height !== undefined ),
      'Should not call toCanvas on a Node with empty bounds, unless all dimensions are provided' );

    x = x !== undefined ? x : Math.ceil( padding - bounds.minX );
    y = y !== undefined ? y : Math.ceil( padding - bounds.minY );
    width = width !== undefined ? width : Math.ceil( bounds.getWidth() + 2 * padding );
    height = height !== undefined ? height : Math.ceil( bounds.getHeight() + 2 * padding );

    const canvas = document.createElement( 'canvas' );
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext( '2d' )!;

    // shift our rendering over by the desired amount
    context.translate( x, y );

    // for API compatibility, we apply our own transform here
    this._transform.getMatrix().canvasAppendTransform( context );

    const wrapper = new CanvasContextWrapper( canvas, context );

    this.renderToCanvasSubtree( wrapper, Matrix3.translation( x, y ).timesMatrix( this._transform.getMatrix() ) );

    callback( canvas, x, y, width, height ); // we used to be asynchronous
  }

  /**
   * Renders this Node to a Canvas, then calls the callback with the data URI from it.
   *
   * @param callback - callback( dataURI {string}, x, y, width, height ) is called, where x,y are computed if not specified.
   * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param [width] - The width of the Canvas output
   * @param [height] - The height of the Canvas output
   */
  public toDataURL( callback: ( dataURI: string, x: number, y: number, width: number, height: number ) => void, x?: number, y?: number, width?: number, height?: number ): void {
    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    this.toCanvas( ( canvas, x, y, width, height ) => {
      // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
      callback( canvas.toDataURL(), x, y, width, height );
    }, x, y, width, height );
  }

  /**
   * Calls the callback with an HTMLImageElement that contains this Node's subtree's visual form.
   * Will always be asynchronous.
   * @deprecated - Use node.rasterized() for creating a rasterized copy, or generally it's best to get the data
   *               URL instead directly.
   *
   * @param callback - callback( image {HTMLImageElement}, x, y ) is called
   * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param [width] - The width of the Canvas output
   * @param [height] - The height of the Canvas output
   */
  public toImage( callback: ( image: HTMLImageElement, x: number, y: number ) => void, x?: number, y?: number, width?: number, height?: number ): void {

    assert && deprecationWarning( 'Node.toImage() is deprecated, please use Node.rasterized() instead' );

    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    this.toDataURL( ( url, x, y ) => {
      // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
      const img = document.createElement( 'img' );
      img.onload = () => {
        callback( img, x, y );
        try {
          // @ts-expect-error - I believe we need to delete this
          delete img.onload;
        }
        catch( e ) {
          // do nothing
        } // fails on Safari 5.1
      };
      img.src = url;
    }, x, y, width, height );
  }

  /**
   * Calls the callback with an Image Node that contains this Node's subtree's visual form. This is always
   * asynchronous, but the resulting image Node can be used with any back-end (Canvas/WebGL/SVG/etc.)
   * @deprecated - Use node.rasterized() instead (should avoid the asynchronous-ness)
   *
   * @param callback - callback( imageNode {Image} ) is called
   * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param [width] - The width of the Canvas output
   * @param [height] - The height of the Canvas output
   */
  public toImageNodeAsynchronous( callback: ( image: Node ) => void, x?: number, y?: number, width?: number, height?: number ): void {

    assert && deprecationWarning( 'Node.toImageNodeAsyncrhonous() is deprecated, please use Node.rasterized() instead' );

    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    this.toImage( ( image, x, y ) => {
      callback( new Node( { // eslint-disable-line no-html-constructors
        children: [
          new Image( image, { x: -x, y: -y } )
        ]
      } ) );
    }, x, y, width, height );
  }

  /**
   * Creates a Node containing an Image Node that contains this Node's subtree's visual form. This is always
   * synchronous, but the resulting image Node can ONLY used with Canvas/WebGL (NOT SVG).
   * @deprecated - Use node.rasterized() instead, should be mostly equivalent if useCanvas:true is provided.
   *
   * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param [width] - The width of the Canvas output
   * @param [height] - The height of the Canvas output
   */
  public toCanvasNodeSynchronous( x?: number, y?: number, width?: number, height?: number ): Node {

    assert && deprecationWarning( 'Node.toCanvasNodeSynchronous() is deprecated, please use Node.rasterized() instead' );

    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    let result: Node | null = null;
    this.toCanvas( ( canvas, x, y ) => {
      result = new Node( { // eslint-disable-line no-html-constructors
        children: [
          new Image( canvas, { x: -x, y: -y } )
        ]
      } );
    }, x, y, width, height );
    assert && assert( result, 'toCanvasNodeSynchronous requires that the node can be rendered only using Canvas' );
    return result!;
  }

  /**
   * Returns an Image that renders this Node. This is always synchronous, and sets initialWidth/initialHeight so that
   * we have the bounds immediately.  Use this method if you need to reduce the number of parent Nodes.
   *
   * NOTE: the resultant Image should be positioned using its bounds rather than (x,y).  To create a Node that can be
   * positioned like any other node, please use toDataURLNodeSynchronous.
   * @deprecated - Use node.rasterized() instead, should be mostly equivalent if wrap:false is provided.
   *
   * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param [width] - The width of the Canvas output
   * @param [height] - The height of the Canvas output
   */
  public toDataURLImageSynchronous( x?: number, y?: number, width?: number, height?: number ): Image {

    assert && deprecationWarning( 'Node.toDataURLImageSychronous() is deprecated, please use Node.rasterized() instead' );

    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    let result: Image | null = null;
    this.toDataURL( ( dataURL, x, y, width, height ) => {
      result = new Image( dataURL, { x: -x, y: -y, initialWidth: width, initialHeight: height } );
    }, x, y, width, height );
    assert && assert( result, 'toDataURL failed to return a result synchronously' );
    return result!;
  }

  /**
   * Returns a Node that contains this Node's subtree's visual form. This is always synchronous, and sets
   * initialWidth/initialHeight so that we have the bounds immediately.  An extra wrapper Node is provided
   * so that transforms can be done independently.  Use this method if you need to be able to transform the node
   * the same way as if it had not been rasterized.
   * @deprecated - Use node.rasterized() instead, should be mostly equivalent
   *
   * @param [x] - The X offset for where the upper-left of the content drawn into the Canvas
   * @param [y] - The Y offset for where the upper-left of the content drawn into the Canvas
   * @param [width] - The width of the Canvas output
   * @param [height] - The height of the Canvas output
   */
  public toDataURLNodeSynchronous( x?: number, y?: number, width?: number, height?: number ): Node {

    assert && deprecationWarning( 'Node.toDataURLNodeSynchronous() is deprecated, please use Node.rasterized() instead' );

    assert && assert( x === undefined || typeof x === 'number', 'If provided, x should be a number' );
    assert && assert( y === undefined || typeof y === 'number', 'If provided, y should be a number' );
    assert && assert( width === undefined || ( typeof width === 'number' && width >= 0 && ( width % 1 === 0 ) ),
      'If provided, width should be a non-negative integer' );
    assert && assert( height === undefined || ( typeof height === 'number' && height >= 0 && ( height % 1 === 0 ) ),
      'If provided, height should be a non-negative integer' );

    return new Node( { // eslint-disable-line no-html-constructors
      children: [
        this.toDataURLImageSynchronous( x, y, width, height )
      ]
    } );
  }

  /**
   * Returns a Node (backed by a scenery Image) that is a rasterized version of this node.
   *
   * @param [options] - See below options. This is also passed directly to the created Image object.
   */
  public rasterized( providedOptions?: RasterizedOptions ): Node {
    const options = optionize<RasterizedOptions, RasterizedOptions>()( {
      // {number} - Controls the resolution of the image relative to the local view units. For example, if our Node is
      // ~100 view units across (in the local coordinate frame) but you want the image to actually have a ~200-pixel
      // resolution, provide resolution:2.
      resolution: 1,

      // {Bounds2|null} - If provided, it will control the x/y/width/height of the toCanvas call. See toCanvas for
      // details on how this controls the rasterization. This is in the "parent" coordinate frame, similar to
      // node.bounds.
      sourceBounds: null,

      // {boolean} - If true, the localBounds of the result will be set in a way such that it will precisely match
      // the visible bounds of the original Node (this). Note that antialiased content (with a much lower resolution)
      // may somewhat spill outside these bounds if this is set to true. Usually this is fine and should be the
      // recommended option. If sourceBounds are provided, they will restrict the used bounds (so it will just
      // represent the bounds of the sliced part of the image).
      useTargetBounds: true,

      // {boolean} - If true, the created Image Node gets wrapped in an extra Node so that it can be transformed
      // independently. If there is no need to transform the resulting node, wrap:false can be passed so that no extra
      // Node is created.
      wrap: true,

      // {boolean} - If true, it will directly use the <canvas> element (only works with canvas/webgl renderers)
      // instead of converting this into a form that can be used with any renderer. May have slightly better
      // performance if svg/dom renderers do not need to be used.
      useCanvas: false,

      // To be passed to the Image node created from the rasterization. See below for options that will override
      // what is passed in.
      imageOptions: {}
    }, providedOptions );

    const resolution = options.resolution;
    const sourceBounds = options.sourceBounds;

    if ( assert ) {
      assert( typeof resolution === 'number' && resolution > 0, 'resolution should be a positive number' );
      assert( sourceBounds === null || sourceBounds instanceof Bounds2, 'sourceBounds should be null or a Bounds2' );
      if ( sourceBounds ) {
        assert( sourceBounds.isValid(), 'sourceBounds should be valid (finite non-negative)' );
        assert( Number.isInteger( sourceBounds.width ), 'sourceBounds.width should be an integer' );
        assert( Number.isInteger( sourceBounds.height ), 'sourceBounds.height should be an integer' );
      }
    }

    // We'll need to wrap it in a container Node temporarily (while rasterizing) for the scale
    const wrapperNode = new Node( { // eslint-disable-line no-html-constructors
      scale: resolution,
      children: [ this ]
    } );

    let transformedBounds = sourceBounds || this.getSafeTransformedVisibleBounds().dilated( 2 ).roundedOut();

    // Unfortunately if we provide a resolution AND bounds, we can't use the source bounds directly.
    if ( resolution !== 1 ) {
      transformedBounds = new Bounds2(
        resolution * transformedBounds.minX,
        resolution * transformedBounds.minY,
        resolution * transformedBounds.maxX,
        resolution * transformedBounds.maxY
      );
      // Compensate for non-integral transformedBounds after our resolution transform
      if ( transformedBounds.width % 1 !== 0 ) {
        transformedBounds.maxX += 1 - ( transformedBounds.width % 1 );
      }
      if ( transformedBounds.height % 1 !== 0 ) {
        transformedBounds.maxY += 1 - ( transformedBounds.height % 1 );
      }
    }

    let image: Image | null = null;

    // NOTE: This callback is executed SYNCHRONOUSLY
    function callback( canvas: HTMLCanvasElement, x: number, y: number, width: number, height: number ): void {
      const imageSource = options.useCanvas ? canvas : canvas.toDataURL();

      image = new Image( imageSource, combineOptions<ImageOptions>( {}, options.imageOptions, {
        x: -x,
        y: -y,
        initialWidth: width,
        initialHeight: height
      } ) );

      // We need to prepend the scale due to order of operations
      image.scale( 1 / resolution, 1 / resolution, true );
    }

    // NOTE: Rounding necessary due to floating point arithmetic in the width/height computation of the bounds
    wrapperNode.toCanvas( callback, -transformedBounds.minX, -transformedBounds.minY, Utils.roundSymmetric( transformedBounds.width ), Utils.roundSymmetric( transformedBounds.height ) );

    assert && assert( image, 'The toCanvas should have executed synchronously' );

    wrapperNode.dispose();

    // For our useTargetBounds option, we do NOT want to include any "safe" bounds, and instead want to stay true to
    // the original bounds. We do filter out invisible subtrees to set the bounds.
    let finalParentBounds = this.getVisibleBounds();
    if ( sourceBounds ) {
      // If we provide sourceBounds, don't have resulting bounds that go outside.
      finalParentBounds = sourceBounds.intersection( finalParentBounds );
    }

    if ( options.useTargetBounds ) {
      image!.imageBounds = image!.parentToLocalBounds( finalParentBounds );
    }

    if ( options.wrap ) {
      const wrappedNode = new Node( { children: [ image! ] } ); // eslint-disable-line no-html-constructors
      if ( options.useTargetBounds ) {
        wrappedNode.localBounds = finalParentBounds;
      }
      return wrappedNode;
    }
    else {
      if ( options.useTargetBounds ) {
        image!.localBounds = image!.parentToLocalBounds( finalParentBounds );
      }
      return image!;
    }
  }

  /**
   * Creates a DOM drawable for this Node's self representation. (scenery-internal)
   *
   * Implemented by subtypes that support DOM self drawables. There is no need to implement this for subtypes that
   * do not allow the DOM renderer (not set in its rendererBitmask).
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public createDOMDrawable( renderer: number, instance: Instance ): DOMSelfDrawable {
    throw new Error( 'createDOMDrawable is abstract. The subtype should either override this method, or not support the DOM renderer' );
  }

  /**
   * Creates an SVG drawable for this Node's self representation. (scenery-internal)
   *
   * Implemented by subtypes that support SVG self drawables. There is no need to implement this for subtypes that
   * do not allow the SVG renderer (not set in its rendererBitmask).
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public createSVGDrawable( renderer: number, instance: Instance ): SVGSelfDrawable {
    throw new Error( 'createSVGDrawable is abstract. The subtype should either override this method, or not support the DOM renderer' );
  }

  /**
   * Creates a Canvas drawable for this Node's self representation. (scenery-internal)
   *
   * Implemented by subtypes that support Canvas self drawables. There is no need to implement this for subtypes that
   * do not allow the Canvas renderer (not set in its rendererBitmask).
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public createCanvasDrawable( renderer: number, instance: Instance ): CanvasSelfDrawable {
    throw new Error( 'createCanvasDrawable is abstract. The subtype should either override this method, or not support the DOM renderer' );
  }

  /**
   * Creates a WebGL drawable for this Node's self representation. (scenery-internal)
   *
   * Implemented by subtypes that support WebGL self drawables. There is no need to implement this for subtypes that
   * do not allow the WebGL renderer (not set in its rendererBitmask).
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public createWebGLDrawable( renderer: number, instance: Instance ): WebGLSelfDrawable {
    throw new Error( 'createWebGLDrawable is abstract. The subtype should either override this method, or not support the DOM renderer' );
  }

  /*---------------------------------------------------------------------------*
   * Instance handling
   *----------------------------------------------------------------------------*/

  /**
   * Returns a reference to the instances array. (scenery-internal)
   */
  public getInstances(): Instance[] {
    return this._instances;
  }

  /**
   * See getInstances() for more information (scenery-internal)
   */
  public get instances(): Instance[] {
    return this.getInstances();
  }

  /**
   * Adds an Instance reference to our array. (scenery-internal)
   */
  public addInstance( instance: Instance ): void {
    this._instances.push( instance );

    this.changedInstanceEmitter.emit( instance, true );
  }

  /**
   * Removes an Instance reference from our array. (scenery-internal)
   */
  public removeInstance( instance: Instance ): void {
    const index = _.indexOf( this._instances, instance );
    assert && assert( index !== -1, 'Cannot remove a Instance from a Node if it was not there' );
    this._instances.splice( index, 1 );

    this.changedInstanceEmitter.emit( instance, false );
  }

  /**
   * Returns whether this Node was visually rendered/displayed by any Display in the last updateDisplay() call. Note
   * that something can be independently displayed visually, and in the PDOM; this method only checks visually.
   *
   * @param [display] - if provided, only check if was visible on this particular Display
   */
  public wasVisuallyDisplayed( display?: Display ): boolean {
    for ( let i = 0; i < this._instances.length; i++ ) {
      const instance = this._instances[ i ];

      // If no display is provided, any instance visibility is enough to be visually displayed
      if ( instance.visible && ( !display || instance.display === display ) ) {
        return true;
      }
    }
    return false;
  }

  /*---------------------------------------------------------------------------*
   * Display handling
   *----------------------------------------------------------------------------*/

  /**
   * Returns a reference to the display array. (scenery-internal)
   */
  public getRootedDisplays(): Display[] {
    return this._rootedDisplays;
  }

  /**
   * See getRootedDisplays() for more information (scenery-internal)
   */
  public get rootedDisplays(): Display[] {
    return this.getRootedDisplays();
  }

  /**
   * Adds an display reference to our array. (scenery-internal)
   */
  public addRootedDisplay( display: Display ): void {
    this._rootedDisplays.push( display );

    // Defined in ParallelDOM.js
    this._pdomDisplaysInfo.onAddedRootedDisplay( display );
  }

  /**
   * Removes a Display reference from our array. (scenery-internal)
   */
  public removeRootedDisplay( display: Display ): void {
    const index = _.indexOf( this._rootedDisplays, display );
    assert && assert( index !== -1, 'Cannot remove a Display from a Node if it was not there' );
    this._rootedDisplays.splice( index, 1 );

    // Defined in ParallelDOM.js
    this._pdomDisplaysInfo.onRemovedRootedDisplay( display );
  }

  private getRecursiveConnectedDisplays( displays: Display[] ): Display[] {
    if ( this.rootedDisplays.length ) {
      displays.push( ...this.rootedDisplays );
    }

    for ( let i = 0; i < this._parents.length; i++ ) {
      displays.push( ...this._parents[ i ].getRecursiveConnectedDisplays( displays ) );
    }

    // do not allow duplicate Displays to get collected infinitely
    return _.uniq( displays );
  }

  /**
   * Get a list of the displays that are connected to this Node. Gathered by looking up the scene graph ancestors and
   * collected all rooted Displays along the way.
   */
  public getConnectedDisplays(): Display[] {
    return _.uniq( this.getRecursiveConnectedDisplays( [] ) );
  }

  /*---------------------------------------------------------------------------*
   * Coordinate transform methods
   *----------------------------------------------------------------------------*/

  /**
   * Returns a point transformed from our local coordinate frame into our parent coordinate frame. Applies our node's
   * transform to it.
   */
  public localToParentPoint( point: Vector2 ): Vector2 {
    return this._transform.transformPosition2( point );
  }

  /**
   * Returns bounds transformed from our local coordinate frame into our parent coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   */
  public localToParentBounds( bounds: Bounds2 ): Bounds2 {
    return this._transform.transformBounds2( bounds );
  }

  /**
   * Returns a point transformed from our parent coordinate frame into our local coordinate frame. Applies the inverse
   * of our node's transform to it.
   */
  public parentToLocalPoint( point: Vector2 ): Vector2 {
    return this._transform.inversePosition2( point );
  }

  /**
   * Returns bounds transformed from our parent coordinate frame into our local coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   */
  public parentToLocalBounds( bounds: Bounds2 ): Bounds2 {
    return this._transform.inverseBounds2( bounds );
  }

  /**
   * A mutable-optimized form of localToParentBounds() that will modify the provided bounds, transforming it from our
   * local coordinate frame to our parent coordinate frame.
   * @returns - The same bounds object.
   */
  public transformBoundsFromLocalToParent( bounds: Bounds2 ): Bounds2 {
    return bounds.transform( this._transform.getMatrix() );
  }

  /**
   * A mutable-optimized form of parentToLocalBounds() that will modify the provided bounds, transforming it from our
   * parent coordinate frame to our local coordinate frame.
   * @returns - The same bounds object.
   */
  public transformBoundsFromParentToLocal( bounds: Bounds2 ): Bounds2 {
    return bounds.transform( this._transform.getInverse() );
  }

  /**
   * Returns a new matrix (fresh copy) that would transform points from our local coordinate frame to the global
   * coordinate frame.
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public getLocalToGlobalMatrix(): Matrix3 {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let node: Node = this; // eslint-disable-line consistent-this

    // we need to apply the transformations in the reverse order, so we temporarily store them
    const matrices = [];

    // concatenation like this has been faster than getting a unique trail, getting its transform, and applying it
    while ( node ) {
      matrices.push( node._transform.getMatrix() );
      assert && assert( node._parents[ 1 ] === undefined, 'getLocalToGlobalMatrix unable to work for DAG' );
      node = node._parents[ 0 ];
    }

    const matrix = Matrix3.identity(); // will be modified in place

    // iterate from the back forwards (from the root Node to here)
    for ( let i = matrices.length - 1; i >= 0; i-- ) {
      matrix.multiplyMatrix( matrices[ i ] );
    }

    // NOTE: always return a fresh copy, getGlobalToLocalMatrix depends on it to minimize instance usage!
    return matrix;
  }

  /**
   * Returns a Transform3 that would transform things from our local coordinate frame to the global coordinate frame.
   * Equivalent to getUniqueTrail().getTransform(), but faster.
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public getUniqueTransform(): Transform3 {
    return new Transform3( this.getLocalToGlobalMatrix() );
  }

  /**
   * Returns a new matrix (fresh copy) that would transform points from the global coordinate frame to our local
   * coordinate frame.
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public getGlobalToLocalMatrix(): Matrix3 {
    return this.getLocalToGlobalMatrix().invert();
  }

  /**
   * Transforms a point from our local coordinate frame to the global coordinate frame.
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public localToGlobalPoint( point: Vector2 ): Vector2 {

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let node: Node = this; // eslint-disable-line consistent-this
    const resultPoint = point.copy();
    while ( node ) {
      // in-place multiplication
      node._transform.getMatrix().multiplyVector2( resultPoint );
      assert && assert( node._parents[ 1 ] === undefined, 'localToGlobalPoint unable to work for DAG' );
      node = node._parents[ 0 ];
    }
    return resultPoint;
  }

  /**
   * Transforms a point from the global coordinate frame to our local coordinate frame.
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public globalToLocalPoint( point: Vector2 ): Vector2 {

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    let node: Node = this; // eslint-disable-line consistent-this
    // TODO: performance: test whether it is faster to get a total transform and then invert (won't compute individual inverses)

    // we need to apply the transformations in the reverse order, so we temporarily store them
    const transforms = [];
    while ( node ) {
      transforms.push( node._transform );
      assert && assert( node._parents[ 1 ] === undefined, 'globalToLocalPoint unable to work for DAG' );
      node = node._parents[ 0 ];
    }

    // iterate from the back forwards (from the root Node to here)
    const resultPoint = point.copy();
    for ( let i = transforms.length - 1; i >= 0; i-- ) {
      // in-place multiplication
      transforms[ i ].getInverse().multiplyVector2( resultPoint );
    }
    return resultPoint;
  }

  /**
   * Transforms bounds from our local coordinate frame to the global coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public localToGlobalBounds( bounds: Bounds2 ): Bounds2 {
    // apply the bounds transform only once, so we can minimize the expansion encountered from multiple rotations
    // it also seems to be a bit faster this way
    return bounds.transformed( this.getLocalToGlobalMatrix() );
  }

  /**
   * Transforms bounds from the global coordinate frame to our local coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public globalToLocalBounds( bounds: Bounds2 ): Bounds2 {
    // apply the bounds transform only once, so we can minimize the expansion encountered from multiple rotations
    return bounds.transformed( this.getGlobalToLocalMatrix() );
  }

  /**
   * Transforms a point from our parent coordinate frame to the global coordinate frame.
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public parentToGlobalPoint( point: Vector2 ): Vector2 {
    assert && assert( this.parents.length <= 1, 'parentToGlobalPoint unable to work for DAG' );
    return this.parents.length ? this.parents[ 0 ].localToGlobalPoint( point ) : point;
  }

  /**
   * Transforms bounds from our parent coordinate frame to the global coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public parentToGlobalBounds( bounds: Bounds2 ): Bounds2 {
    assert && assert( this.parents.length <= 1, 'parentToGlobalBounds unable to work for DAG' );
    return this.parents.length ? this.parents[ 0 ].localToGlobalBounds( bounds ) : bounds;
  }

  /**
   * Transforms a point from the global coordinate frame to our parent coordinate frame.
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public globalToParentPoint( point: Vector2 ): Vector2 {
    assert && assert( this.parents.length <= 1, 'globalToParentPoint unable to work for DAG' );
    return this.parents.length ? this.parents[ 0 ].globalToLocalPoint( point ) : point;
  }

  /**
   * Transforms bounds from the global coordinate frame to our parent coordinate frame. If it includes a
   * rotation, the resulting bounding box will include every point that could have been in the original bounding box
   * (and it can be expanded).
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   */
  public globalToParentBounds( bounds: Bounds2 ): Bounds2 {
    assert && assert( this.parents.length <= 1, 'globalToParentBounds unable to work for DAG' );
    return this.parents.length ? this.parents[ 0 ].globalToLocalBounds( bounds ) : bounds;
  }

  /**
   * Returns a bounding box for this Node (and its sub-tree) in the global coordinate frame.
   *
   * NOTE: If there are multiple instances of this Node (e.g. this or one ancestor has two parents), it will fail
   * with an assertion (since the transform wouldn't be uniquely defined).
   *
   * NOTE: This requires computation of this node's subtree bounds, which may incur some performance loss.
   */
  public getGlobalBounds(): Bounds2 {
    assert && assert( this.parents.length <= 1, 'globalBounds unable to work for DAG' );
    return this.parentToGlobalBounds( this.getBounds() );
  }

  /**
   * See getGlobalBounds() for more information
   */
  public get globalBounds(): Bounds2 {
    return this.getGlobalBounds();
  }

  /**
   * Returns the bounds of any other Node in our local coordinate frame.
   *
   * NOTE: If this node or the passed in Node have multiple instances (e.g. this or one ancestor has two parents), it will fail
   * with an assertion.
   *
   * TODO: Possible to be well-defined and have multiple instances of each.
   */
  public boundsOf( node: Node ): Bounds2 {
    return this.globalToLocalBounds( node.getGlobalBounds() );
  }

  /**
   * Returns the bounds of this Node in another node's local coordinate frame.
   *
   * NOTE: If this node or the passed in Node have multiple instances (e.g. this or one ancestor has two parents), it will fail
   * with an assertion.
   *
   * TODO: Possible to be well-defined and have multiple instances of each.
   */
  public boundsTo( node: Node ): Bounds2 {
    return node.globalToLocalBounds( this.getGlobalBounds() );
  }

  /*---------------------------------------------------------------------------*
   * Drawable handling
   *----------------------------------------------------------------------------*/

  /**
   * Adds the drawable to our list of drawables to notify of visual changes. (scenery-internal)
   */
  public attachDrawable( drawable: Drawable ): this {
    this._drawables.push( drawable );
    return this; // allow chaining
  }

  /**
   * Removes the drawable from our list of drawables to notify of visual changes. (scenery-internal)
   */
  public detachDrawable( drawable: Drawable ): this {
    const index = _.indexOf( this._drawables, drawable );

    assert && assert( index >= 0, 'Invalid operation: trying to detach a non-referenced drawable' );

    this._drawables.splice( index, 1 ); // TODO: replace with a remove() function
    return this;
  }

  /**
   * Scans the options object for key names that correspond to ES5 setters or other setter functions, and calls those
   * with the values.
   *
   * For example:
   *
   * node.mutate( { top: 0, left: 5 } );
   *
   * will be equivalent to:
   *
   * node.left = 5;
   * node.top = 0;
   *
   * In particular, note that the order is different. Mutators will be applied in the order of _mutatorKeys, which can
   * be added to by subtypes.
   *
   * Additionally, some keys are actually direct function names, like 'scale'. mutate( { scale: 2 } ) will call
   * node.scale( 2 ) instead of activating an ES5 setter directly.
   */
  public mutate( options?: NodeOptions ): this {

    if ( !options ) {
      return this;
    }

    assert && assert( Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    // @ts-expect-error
    assert && assert( _.filter( [ 'translation', 'x', 'left', 'right', 'centerX', 'centerTop', 'rightTop', 'leftCenter', 'center', 'rightCenter', 'leftBottom', 'centerBottom', 'rightBottom' ], key => options[ key ] !== undefined ).length <= 1,
      `More than one mutation on this Node set the x component, check ${Object.keys( options ).join( ',' )}` );

    // @ts-expect-error
    assert && assert( _.filter( [ 'translation', 'y', 'top', 'bottom', 'centerY', 'centerTop', 'rightTop', 'leftCenter', 'center', 'rightCenter', 'leftBottom', 'centerBottom', 'rightBottom' ], key => options[ key ] !== undefined ).length <= 1,
      `More than one mutation on this Node set the y component, check ${Object.keys( options ).join( ',' )}` );

    if ( assert && options.hasOwnProperty( 'enabled' ) && options.hasOwnProperty( 'enabledProperty' ) ) {
      assert && assert( options.enabledProperty!.value === options.enabled, 'If both enabled and enabledProperty are provided, then values should match' );
    }
    if ( assert && options.hasOwnProperty( 'inputEnabled' ) && options.hasOwnProperty( 'inputEnabledProperty' ) ) {
      assert && assert( options.inputEnabledProperty!.value === options.inputEnabled, 'If both inputEnabled and inputEnabledProperty are provided, then values should match' );
    }
    if ( assert && options.hasOwnProperty( 'visible' ) && options.hasOwnProperty( 'visibleProperty' ) ) {
      assert && assert( options.visibleProperty!.value === options.visible, 'If both visible and visibleProperty are provided, then values should match' );
    }
    if ( assert && options.hasOwnProperty( 'pickable' ) && options.hasOwnProperty( 'pickableProperty' ) ) {
      assert && assert( options.pickableProperty!.value === options.pickable, 'If both pickable and pickableProperty are provided, then values should match' );
    }

    _.each( this._mutatorKeys, key => {

      // See https://github.com/phetsims/scenery/issues/580 for more about passing undefined.
      // @ts-expect-error
      assert && assert( !options.hasOwnProperty( key ) || options[ key ] !== undefined, `Undefined not allowed for Node key: ${key}` );

      // @ts-expect-error - Hmm, better way to check this?
      if ( options[ key ] !== undefined ) {
        const descriptor = Object.getOwnPropertyDescriptor( Node.prototype, key );

        // if the key refers to a function that is not ES5 writable, it will execute that function with the single argument
        if ( descriptor && typeof descriptor.value === 'function' ) {
          // @ts-expect-error
          this[ key ]( options[ key ] );
        }
        else {
          // @ts-expect-error
          this[ key ] = options[ key ];
        }
      }
    } );

    this.initializePhetioObject( { phetioType: Node.NodeIO, phetioState: PHET_IO_STATE_DEFAULT }, options );

    return this; // allow chaining
  }

  protected override initializePhetioObject( baseOptions: Partial<PhetioObjectOptions>, config: NodeOptions ): void {

    // Track this, so we only override our visibleProperty once.
    const wasInstrumented = this.isPhetioInstrumented();

    super.initializePhetioObject( baseOptions, config );

    if ( Tandem.PHET_IO_ENABLED && !wasInstrumented && this.isPhetioInstrumented() ) {

      // For each supported TinyForwardingProperty, if a Property was already specified in the options (in the
      // constructor or mutate), then it will be set as this.targetProperty there. Here we only create the default
      // instrumented one if another hasn't already been specified.

      this._visibleProperty.initializePhetio( this, VISIBLE_PROPERTY_TANDEM_NAME, () => new BooleanProperty( this.visible, combineOptions<BooleanPropertyOptions>( {

          // by default, use the value from the Node
          phetioReadOnly: this.phetioReadOnly,
          tandem: this.tandem.createTandem( VISIBLE_PROPERTY_TANDEM_NAME ),
          phetioDocumentation: 'Controls whether the Node will be visible (and interactive).'
        }, config.visiblePropertyOptions ) )
      );

      this._enabledProperty.initializePhetio( this, ENABLED_PROPERTY_TANDEM_NAME, () => new EnabledProperty( this.enabled, combineOptions<EnabledPropertyOptions>( {

          // by default, use the value from the Node
          phetioReadOnly: this.phetioReadOnly,
          phetioDocumentation: 'Sets whether the node is enabled. This will set whether input is enabled for this Node and ' +
                               'most often children as well. It will also control and toggle the "disabled look" of the node.',
          tandem: this.tandem.createTandem( ENABLED_PROPERTY_TANDEM_NAME )
        }, config.enabledPropertyOptions ) )
      );

      this._inputEnabledProperty.initializePhetio( this, INPUT_ENABLED_PROPERTY_TANDEM_NAME, () => new Property( this.inputEnabled, combineOptions<PropertyOptions<boolean>>( {

          // by default, use the value from the Node
          phetioReadOnly: this.phetioReadOnly,
          tandem: this.tandem.createTandem( INPUT_ENABLED_PROPERTY_TANDEM_NAME ),
          phetioValueType: BooleanIO,
          phetioFeatured: true, // Since this property is opt-in, we typically only opt-in when it should be featured
          phetioDocumentation: 'Sets whether the element will have input enabled, and hence be interactive.'
        }, config.inputEnabledPropertyOptions ) )
      );
    }
  }

  /**
   * Set the visibility of this Node with respect to the Voicing feature. Totally separate from graphical display.
   * When visible, this Node and all of its ancestors will be able to speak with Voicing. When voicingVisible
   * is false, all Voicing under this Node will be muted. `voicingVisible` properties exist in Node.ts because
   * it is useful to set `voicingVisible` on a root that is composed with Voicing.ts. We cannot put all of the
   * Voicing.ts implementation in Node because that would have a massive memory impact. See Voicing.ts for more
   * information.
   */
  public setVoicingVisible( visible: boolean ): void {
    if ( this.voicingVisibleProperty.value !== visible ) {
      this.voicingVisibleProperty.value = visible;
    }
  }

  public set voicingVisible( visible: boolean ) { this.setVoicingVisible( visible ); }

  public get voicingVisible(): boolean { return this.isVoicingVisible(); }

  /**
   * Returns whether this Node is voicingVisible. When true Utterances for this Node can be announced with the
   * Voicing feature, see Voicing.ts for more information.
   */
  public isVoicingVisible(): boolean {
    return this.voicingVisibleProperty.value;
  }

  /**
   * Override for extra information in the debugging output (from Display.getDebugHTML()). (scenery-internal)
   */
  public getDebugHTMLExtras(): string {
    return '';
  }

  /**
   * Makes this Node's subtree available for inspection.
   */
  public inspect(): void {
    localStorage.scenerySnapshot = JSON.stringify( {
      type: 'Subtree',
      rootNodeId: this.id,
      nodes: serializeConnectedNodes( this )
    } );
  }

  /**
   * Returns a debugging string that is an attempted serialization of this node's sub-tree.
   *
   * @param spaces - Whitespace to add
   * @param [includeChildren]
   */
  public override toString( spaces: string, includeChildren?: boolean ): string {
    return `${this.constructor.name}#${this.id}`;
  }

  /**
   * Performs checks to see if the internal state of Instance references is correct at a certain point in/after the
   * Display's updateDisplay(). (scenery-internal)
   */
  public auditInstanceSubtreeForDisplay( display: Display ): void {
    if ( assertSlow ) {
      const numInstances = this._instances.length;
      for ( let i = 0; i < numInstances; i++ ) {
        const instance = this._instances[ i ];
        if ( instance.display === display ) {
          assertSlow( instance.trail!.isValid(),
            `Invalid trail on Instance: ${instance.toString()} with trail ${instance.trail!.toString()}` );
        }
      }

      // audit all of the children
      this.children.forEach( child => {
        child.auditInstanceSubtreeForDisplay( display );
      } );
    }
  }

  /**
   * When we add or remove any number of bounds listeners, we want to increment/decrement internal information.
   *
   * @param deltaQuantity - If positive, the number of listeners being added, otherwise the number removed
   */
  private onBoundsListenersAddedOrRemoved( deltaQuantity: number ): void {
    this.changeBoundsEventCount( deltaQuantity );
    this._boundsEventSelfCount += deltaQuantity;
  }

  /**
   * Disposes the node, releasing all references that it maintained.
   */
  public override dispose(): void {

    // remove all PDOM input listeners
    this.disposeParallelDOM();

    // When disposing, remove all children and parents. See https://github.com/phetsims/scenery/issues/629
    this.removeAllChildren();
    this.detach();

    // In opposite order of creation
    this._inputEnabledProperty.dispose();
    this._enabledProperty.dispose();
    this._pickableProperty.dispose();
    this._visibleProperty.dispose();

    // Tear-down in the reverse order Node was created
    super.dispose();
  }

  /**
   * Disposes this Node and all other descendant nodes.
   *
   * NOTE: Use with caution, as you should not re-use any Node touched by this. Not compatible with most DAG
   *       techniques.
   */
  public disposeSubtree(): void {
    if ( !this.isDisposed ) {
      // makes a copy before disposing
      const children = this.children;

      this.dispose();

      for ( let i = 0; i < children.length; i++ ) {
        children[ i ].disposeSubtree();
      }
    }
  }


  /**
   * A default for getTrails() searches, returns whether the Node has no parents.
   */
  public static defaultTrailPredicate( node: Node ): boolean {
    return node._parents.length === 0;
  }

  /**
   * A default for getLeafTrails() searches, returns whether the Node has no parents.
   */
  public static defaultLeafTrailPredicate( node: Node ): boolean {
    return node._children.length === 0;
  }

  public static NodeIO: IOType;

  // A mapping of all of the default options provided to Node
  public static readonly DEFAULT_NODE_OPTIONS = DEFAULT_OPTIONS;

}

Node.prototype._mutatorKeys = ACCESSIBILITY_OPTION_KEYS.concat( NODE_OPTION_KEYS );

/**
 * {Array.<String>} - List of all dirty flags that should be available on drawables created from this Node (or
 *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
 *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
 * (scenery-internal)
 *
 * Should be overridden by subtypes.
 */
Node.prototype.drawableMarkFlags = [];

scenery.register( 'Node', Node );

// {IOType}
Node.NodeIO = new IOType( 'NodeIO', {
  valueType: Node,
  documentation: 'The base type for graphical and potentially interactive objects.',
  metadataDefaults: {
    phetioState: PHET_IO_STATE_DEFAULT
  }
} );

// We use interface extension, so we can't export Node at its declaration location
export default Node;