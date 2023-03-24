// Copyright 2021-2023, University of Colorado Boulder

/**
 * The main type interface for input listeners
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import { SceneryEvent } from '../imports.js';

export type SceneryListenerFunction<T extends Event = Event> = ( event: SceneryEvent<T> ) => void;

type TInputListener = {
  interrupt?: () => void;
  cursor?: string | null;

  // Only applies to globalkeydown/globalkeyup. When true, this listener is fired during the 'capture'
  // phase. Listeners are fired BEFORE the dispatch through the scene graph. (very similar to DOM addEventListener's
  // useCapture).
  capture?: boolean;

  listener?: unknown;

  // Function that returns the Bounds2 for AnimatedPanZoomListener to keep in view during drag input.
  // Bounds are in the global coordinate frame.
  // While dragging, the AnimatedPanZoomListener will try to keep these bounds in view. Intended to be
  // called from a listener attached to a Pointer so that the API is compatible with multi-touch.
  createPanTargetBounds?: ( () => Bounds2 ) | null;

  ////////////////////////////////////////////////
  //////////////////////////////////////////////
  // Only actual events below here

  focus?: SceneryListenerFunction<FocusEvent>;
  blur?: SceneryListenerFunction<FocusEvent>;
  focusin?: SceneryListenerFunction<FocusEvent>;
  focusout?: SceneryListenerFunction<FocusEvent>;

  keydown?: SceneryListenerFunction<KeyboardEvent>;
  keyup?: SceneryListenerFunction<KeyboardEvent>;

  globalkeydown?: SceneryListenerFunction<KeyboardEvent>;
  globalkeyup?: SceneryListenerFunction<KeyboardEvent>;

  click?: SceneryListenerFunction<MouseEvent>;
  input?: SceneryListenerFunction<Event | InputEvent>;
  change?: SceneryListenerFunction;

  down?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mousedown?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchdown?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  pendown?: SceneryListenerFunction<PointerEvent>;

  up?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mouseup?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchup?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penup?: SceneryListenerFunction<PointerEvent>;

  cancel?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  mousecancel?: SceneryListenerFunction<PointerEvent>;
  touchcancel?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  pencancel?: SceneryListenerFunction<PointerEvent>;

  move?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mousemove?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchmove?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penmove?: SceneryListenerFunction<PointerEvent>;

  wheel?: SceneryListenerFunction<WheelEvent>;
  mousewheel?: SceneryListenerFunction<WheelEvent>;
  touchwheel?: SceneryListenerFunction<WheelEvent>;
  penwheel?: SceneryListenerFunction<WheelEvent>;

  enter?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mouseenter?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchenter?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penenter?: SceneryListenerFunction<PointerEvent>;

  exit?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mouseexit?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchexit?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penexit?: SceneryListenerFunction<PointerEvent>;

  over?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mouseover?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchover?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penover?: SceneryListenerFunction<PointerEvent>;

  out?: SceneryListenerFunction<MouseEvent | TouchEvent | PointerEvent>;
  mouseout?: SceneryListenerFunction<MouseEvent | PointerEvent>;
  touchout?: SceneryListenerFunction<TouchEvent | PointerEvent>;
  penout?: SceneryListenerFunction<PointerEvent>;
};

// Exclude all but the actual browser events
export type SupportedEventTypes = keyof StrictOmit<TInputListener, 'interrupt' | 'cursor' | 'capture' | 'listener' | 'createPanTargetBounds'>;

export default TInputListener;
